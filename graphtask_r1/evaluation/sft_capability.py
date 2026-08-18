from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.evaluation.kqapro_val import (
    CompletionClient,
    KQAProModelConfig,
    OpenAICompletionClient,
)
from graphtask_r1.generation import verbalize
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import execute_graphscript, parse_graphscript
from graphtask_r1.training.ms_swift_data import convert_rl_row
from graphtask_r1.training.parsing import parse_task_proposal
from graphtask_r1.utils import write_json, write_records

LOGGER = logging.getLogger(__name__)
RewardScorer = Callable[
    [str, str, str, dict[str, Any] | None], Awaitable[dict[str, float]]
]

STAGE_LABELS = {
    0: "非 JSON",
    1: "额外文本",
    2: "Schema / 版本",
    3: "结构 / 契约",
    4: "执行失败",
    5: "认证失败",
    6: "认证成功",
}


class SFTCapabilityConfig(BaseModel):
    """Bounded, replayable probe for a mixed-role SFT checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: KQAProModelConfig
    questioner_input: Path
    solver_input: Path
    opponent_url: str = Field(min_length=1)
    groups: int = Field(default=3, ge=1, le=20)
    candidates_per_role: int = Field(default=4, ge=1, le=8)
    opponent_samples: int = Field(default=4, ge=1, le=16)
    seed: int = 42
    round_index: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    request_timeout_s: float = Field(default=600.0, gt=0.0)
    request_retries: int = Field(default=1, ge=0, le=10)


class ProbeClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        trace_id: str,
        seed: int,
    ) -> Any: ...

    def flush(self) -> None: ...


def load_sft_capability_config(path: Path) -> SFTCapabilityConfig:
    raw = yaml.safe_load(os.path.expandvars(path.read_text()))
    if not isinstance(raw, dict):
        raise ValueError(f"SFT capability config must be a mapping: {path}")
    return SFTCapabilityConfig.model_validate(raw)


def _sample_role_rows(
    path: Path,
    *,
    data_source: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.get_field_index("data_source") < 0:
        raise ValueError(f"probe input must be an RL parquet with data_source: {path}")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    eligible = 0
    for batch in parquet.iter_batches(batch_size=64):
        for raw in batch.to_pylist():
            if str(raw.get("data_source", "")) != data_source:
                continue
            eligible += 1
            row = cast(dict[str, Any], raw)
            if len(selected) < count:
                selected.append(row)
                continue
            replacement = rng.randrange(eligible)
            if replacement < count:
                selected[replacement] = row
    if len(selected) < count:
        raise ValueError(
            f"{path} has {eligible} rows for {data_source}; {count} groups requested"
        )
    rng.shuffle(selected)
    return selected


def _normalized_probe_row(
    row: Mapping[str, Any],
    *,
    config: SFTCapabilityConfig,
) -> dict[str, Any]:
    converted = convert_rl_row(row)
    messages = converted["messages"]
    extra_info = converted["extra_info"]
    if not isinstance(messages, list) or not isinstance(extra_info, dict):
        raise ValueError("converted RL probe row has invalid messages or extra_info")
    source = str(converted["data_source"])
    if source == "graphtask/questioner":
        extra_info = {
            **extra_info,
            "opponent_url": config.opponent_url,
            "opponent_samples": config.opponent_samples,
            "round": config.round_index,
        }
    return {
        "messages": messages,
        "data_source": source,
        "ground_truth": str(converted["ground_truth"]),
        "extra_info": extra_info,
        "uid": str(converted["uid"]),
    }


def _reason_codes(components: Mapping[str, float]) -> list[str]:
    return sorted(
        name.removeprefix("reject_").upper()
        for name, value in components.items()
        if name.startswith("reject_") and float(value) > 0.0
    )


def _questioner_artifact(
    completion: str,
    info: Mapping[str, Any],
    backends: dict[str, GraphBackend],
) -> dict[str, Any]:
    snapshot = str(info.get("graph_snapshot", "toy-v1"))
    if snapshot not in backends:
        backends[snapshot] = backend_from_snapshot(snapshot)
    backend = backends[snapshot]
    if str(info.get("interaction_mode", "tool")) == "graphscript":
        script = parse_graphscript(
            completion,
            max_follow_limit=int(info.get("max_follow_limit", 100)),
        )
        topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
        execution = execute_graphscript(
            script,
            backend,
            seed_entity=topic_ids[0] if len(topic_ids) == 1 else None,
            allowed_relations=frozenset(
                str(value) for value in info.get("allowed_relations", [])
            ),
            max_edge_visits=int(info.get("max_edge_visits", 200)),
            max_returned_entities=int(info.get("max_returned_entities", 1_000)),
            trace_id=str(info.get("task_id", "sft-capability-questioner")),
        )
        program = execution.program
        answers = execution.answers
    else:
        proposal = parse_task_proposal(completion)
        program = proposal.program
        answers = backend.execute_program(program)
    return {
        "generated_question": verbalize(program, backend),
        "generated_answers": list(answers.values()),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row["components"].get(key, 0.0)) for row in rows) / len(rows)


def summarize_sft_capability(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        by_role[str(row["role"])].append(row)
    summary: dict[str, Any] = {"candidates": len(results), "roles": {}}
    for role in ("questioner", "solver"):
        rows = by_role.get(role, [])
        reasons = Counter(
            reason for row in rows for reason in cast(Sequence[str], row["reason_codes"])
        )
        values: dict[str, Any] = {
            "candidates": len(rows),
            "mean_score": _mean(rows, "score"),
            "mean_raw_score": _mean(rows, "raw_score"),
            "rejection_reasons": dict(sorted(reasons.items())),
        }
        if role == "questioner":
            stages = Counter(
                int(float(row["components"].get("reward_stage", -1.0))) for row in rows
            )
            values.update(
                {
                    "certified_rate": _mean(rows, "certified"),
                    "stage_counts": {
                        str(stage): stages.get(stage, 0) for stage in range(7)
                    },
                }
            )
        else:
            values.update(
                {
                    "f1": _mean(rows, "f1"),
                    "exact_match": _mean(rows, "exact_match"),
                }
            )
        summary["roles"][role] = values
    return summary


def _score_class(score: float) -> str:
    if score > 0.0:
        return "positive"
    if score < 0.0:
        return "negative"
    return "neutral"


def render_sft_capability_html(
    results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_path: Path,
) -> None:
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in results:
        grouped[int(row["group_index"])][str(row["role"])].append(row)

    summary_cards = []
    roles = cast(Mapping[str, Mapping[str, Any]], summary.get("roles", {}))
    for role, label in (("questioner", "Questioner"), ("solver", "Solver")):
        values = roles.get(role, {})
        extra = (
            f"认证率 {float(values.get('certified_rate', 0.0)):.1%}"
            if role == "questioner"
            else (
                f"F1 {float(values.get('f1', 0.0)):.3f} · "
                f"EM {float(values.get('exact_match', 0.0)):.3f}"
            )
        )
        summary_cards.append(
            "<article class='summary-card'>"
            f"<h2>{label}</h2>"
            f"<strong>{float(values.get('mean_raw_score', 0.0)):.3f}</strong>"
            "<span>平均 raw reward</span>"
            f"<p>{extra}</p>"
            "</article>"
        )

    group_sections = []
    for group_index in sorted(grouped):
        role_sections = []
        for role, label in (("questioner", "Questioner"), ("solver", "Solver")):
            rows = grouped[group_index].get(role, [])
            candidate_cards = []
            for row in rows:
                components = cast(Mapping[str, Any], row["components"])
                raw_score = float(components.get("raw_score", components.get("score", 0.0)))
                stage_value = components.get("reward_stage")
                stage = int(float(stage_value)) if stage_value is not None else None
                stage_text = STAGE_LABELS.get(stage, "—") if stage is not None else "—"
                reasons = ", ".join(cast(Sequence[str], row["reason_codes"])) or "无"
                component_rows = "".join(
                    f"<tr><th>{escape(str(name))}</th><td>{float(value):.6f}</td></tr>"
                    for name, value in sorted(components.items())
                )
                generated_answers = escape(
                    json.dumps(row.get("generated_answers", []), ensure_ascii=False)
                )
                generated_html = (
                    "<div class='generated-question'><b>认证后的规范问题</b>"
                    f"<p>{escape(str(row.get('generated_question', '')))}</p>"
                    "<b>程序执行答案</b>"
                    f"<p>{generated_answers}</p>"
                    "</div>"
                    if row.get("generated_question")
                    else ""
                )
                candidate_cards.append(
                    f"<article class='candidate {_score_class(raw_score)}'>"
                    "<header>"
                    f"<b>候选 {int(row['candidate_index']) + 1}</b>"
                    f"<span>raw {raw_score:.4f}</span>"
                    f"<span>训练分 {float(components.get('score', 0.0)):.4f}</span>"
                    f"<span>阶段 {stage if stage is not None else '—'} · {stage_text}</span>"
                    "</header>"
                    f"<p class='reason'>拒绝原因：{escape(reasons)}</p>"
                    f"{generated_html}"
                    f"<pre>{escape(str(row['completion']))}</pre>"
                    "<details><summary>Reward components</summary>"
                    f"<table>{component_rows}</table></details>"
                    "</article>"
                )
            prompt = rows[0]["messages"] if rows else []
            uid = escape(str(rows[0]["uid"])) if rows else ""
            ground_truth_html = (
                "<details><summary>查看 Solver ground truth</summary>"
                f"<pre>{escape(str(rows[0]['ground_truth']))}</pre></details>"
                if role == "solver" and rows
                else ""
            )
            role_sections.append(
                "<section class='role'>"
                f"<h3>{label} <small>{uid}</small></h3>"
                "<details><summary>查看完整 prompt</summary>"
                f"<pre>{escape(json.dumps(prompt, indent=2, ensure_ascii=False))}</pre>"
                "</details>"
                f"{ground_truth_html}"
                f"{''.join(candidate_cards)}"
                "</section>"
            )
        group_sections.append(
            f"<section class='group'><h2>第 {group_index + 1} 组</h2>"
            f"<div class='roles'>{''.join(role_sections)}</div></section>"
        )

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>SFT 双角色能力探针</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#182033;--muted:#667085;--line:#d9dfeb;
--good:#087a55;--bad:#bd2c2c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);
color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}
h1{{margin:0 0 6px}}.subtitle,small{{color:var(--muted);font-weight:400}}.summaries{{display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:22px 0}}.summary-card,.group,
.role,.candidate{{background:var(--card);border:1px solid var(--line);border-radius:14px}}
.summary-card{{padding:18px}}.summary-card h2,.summary-card p{{margin:0}}
.summary-card strong{{display:block;
font-size:32px}}.summary-card span{{color:var(--muted)}}.group{{padding:18px;margin:18px 0}}
.group>h2{{margin-top:0}}.roles{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.role{{padding:14px;min-width:0}}.role h3{{margin-top:0}}.candidate{{padding:12px;margin-top:12px;
border-left-width:5px}}.candidate.positive{{border-left-color:var(--good)}}.candidate.negative{{border-left-color:var(--bad)}}
.candidate.neutral{{border-left-color:#8a94a6}}header{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
header span{{background:#eef2f8;border-radius:999px;padding:2px 8px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;
background:#111827;color:#e5e7eb;border-radius:9px;padding:12px;max-height:420px;overflow:auto}}
.reason{{color:var(--muted)}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid var(--line);
padding:5px;text-align:left}}@media(max-width:900px){{.roles,.summaries{{grid-template-columns:1fr}}}}
.generated-question{{background:#f0fdf8;border:1px solid #b6ead8;border-radius:9px;padding:10px}}
.generated-question p{{margin:4px 0 9px}}
</style></head><body><main><h1>SFT 后 Questioner / Solver 能力探针</h1>
<p class="subtitle">按组顺序生成；所有分数来自训练使用的真实 reward 链。</p>
<div class="summaries">{''.join(summary_cards)}</div>
{''.join(group_sections)}</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _persist_probe(
    results: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    summary = summarize_sft_capability(results)
    write_json(output_dir / "results.json", list(results))
    write_json(output_dir / "summary.json", summary)
    render_sft_capability_html(results, summary, output_dir / "report.html")
    return summary


async def visualize_sft_capability(
    config: SFTCapabilityConfig,
    output_dir: Path,
    *,
    groups: int | None = None,
    client: CompletionClient | None = None,
    scorer: RewardScorer | None = None,
) -> dict[str, Any]:
    """Generate and score one Questioner/Solver prompt pair at a time."""
    effective_groups = groups or config.groups
    if not 1 <= effective_groups <= 20:
        raise ValueError("groups must be between 1 and 20")
    output_dir.mkdir(parents=True, exist_ok=True)
    questioner_rows = _sample_role_rows(
        config.questioner_input,
        data_source="graphtask/questioner",
        count=effective_groups,
        seed=config.seed + 11,
    )
    solver_rows = _sample_role_rows(
        config.solver_input,
        data_source="graphtask/solver",
        count=effective_groups,
        seed=config.seed + 29,
    )
    owned_client = client is None
    effective_client: ProbeClient = client or OpenAICompletionClient(
        config.model,
        timeout_s=config.request_timeout_s,
        retries=config.request_retries,
        cache_path=output_dir / "cache" / "sft-capability.json",
        temperature=config.temperature,
        top_p=config.top_p,
    )
    if scorer is None:
        from graphtask_r1.training.ms_swift_reward import compute_score

        effective_scorer = cast(RewardScorer, compute_score)
    else:
        effective_scorer = scorer
    results: list[dict[str, Any]] = []
    probe_backends: dict[str, GraphBackend] = {}
    summary: dict[str, Any] = {"candidates": 0, "roles": {}}
    try:
        for group_index, (questioner_raw, solver_raw) in enumerate(
            zip(questioner_rows, solver_rows, strict=True)
        ):
            LOGGER.info(
                "sft_capability_group_started group=%d/%d",
                group_index + 1,
                effective_groups,
            )
            for role_index, (role, raw) in enumerate(
                (("questioner", questioner_raw), ("solver", solver_raw))
            ):
                row = _normalized_probe_row(raw, config=config)
                for candidate_index in range(config.candidates_per_role):
                    generation_seed = (
                        config.seed + group_index * 10_000 + role_index * 1_000 + candidate_index
                    )
                    trace_id = (
                        f"sft-capability:g{group_index:03d}:{role}:c{candidate_index:03d}"
                    )
                    completion = await effective_client.complete(
                        cast(Sequence[Mapping[str, str]], row["messages"]),
                        trace_id=trace_id,
                        seed=generation_seed,
                    )
                    content = str(completion.content)
                    components = await effective_scorer(
                        str(row["data_source"]),
                        content,
                        str(row["ground_truth"]),
                        cast(dict[str, Any], row["extra_info"]),
                    )
                    artifact: dict[str, Any] = {}
                    if role == "questioner" and int(components.get("reward_stage", -1.0)) == 6:
                        artifact = _questioner_artifact(
                            content,
                            cast(dict[str, Any], row["extra_info"]),
                            probe_backends,
                        )
                    record = {
                        "group_index": group_index,
                        "role": role,
                        "uid": row["uid"],
                        "task_id": str(row["extra_info"].get("task_id", "")),
                        "candidate_index": candidate_index,
                        "seed": generation_seed,
                        "trace_id": trace_id,
                        "messages": row["messages"],
                        "ground_truth": row["ground_truth"],
                        "completion": content,
                        "completion_tokens": completion.completion_tokens,
                        "cache_hit": bool(completion.cached),
                        "reason_codes": _reason_codes(components),
                        "components": components,
                        **artifact,
                    }
                    results.append(record)
                    LOGGER.info(
                        "\n=== SFT probe group %d/%d · %s · candidate %d/%d ===\n"
                        "uid=%s raw_reward=%.6f train_reward=%.6f stage=%s reasons=%s\n"
                        "completion:\n%s\ncomponents=%s",
                        group_index + 1,
                        effective_groups,
                        role,
                        candidate_index + 1,
                        config.candidates_per_role,
                        row["uid"],
                        float(components.get("raw_score", components.get("score", 0.0))),
                        float(components.get("score", 0.0)),
                        components.get("reward_stage", "—"),
                        ",".join(record["reason_codes"]) or "none",
                        content,
                        json.dumps(components, sort_keys=True, ensure_ascii=False),
                    )
            summary = _persist_probe(results, output_dir)
            LOGGER.info(
                "sft_capability_group_completed group=%d/%d report=%s",
                group_index + 1,
                effective_groups,
                output_dir / "report.html",
            )
    finally:
        if owned_client:
            effective_client.flush()
        if results:
            summary = _persist_probe(results, output_dir)
    write_records(output_dir / "results.parquet", results)
    return {
        "groups": effective_groups,
        "candidates_per_role": config.candidates_per_role,
        "model": config.model.model,
        "summary": summary,
        "artifacts": {
            "html": str(output_dir / "report.html"),
            "json": str(output_dir / "results.json"),
            "parquet": str(output_dir / "results.parquet"),
            "summary": str(output_dir / "summary.json"),
        },
    }
