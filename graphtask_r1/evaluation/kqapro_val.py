from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import GraphScriptError, execute_graphscript, parse_graphscript
from graphtask_r1.schema import AnswerSet, RelationInfo, TaskCertificate
from graphtask_r1.training.prompts import role_prompt
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.utils import read_json, stable_hash, write_json, write_records

LOGGER = logging.getLogger(__name__)
ModelStage = Literal["base", "sft", "grpo"]


class KQAProModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    max_completion_tokens: int = Field(default=4_096, ge=1, le=40_960)


class KQAProValConfig(BaseModel):
    """Single-checkpoint KQAPro evaluation contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: KQAProModelConfig
    input_path: Path
    relation_catalog: Path
    graph_snapshot: str = "kqapro-v1"
    graphscript_version: Literal["0.3"] = "0.3"
    seed: int = 42
    concurrency: int = Field(default=8, ge=1)
    request_timeout_s: float = Field(default=180.0, gt=0)
    request_retries: int = Field(default=2, ge=0, le=10)
    max_follow_limit: int = Field(default=100, ge=1)
    max_edge_visits: int = Field(default=200, ge=1)
    max_returned_entities: int = Field(default=1_000, ge=1)

class CompletionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    completion_tokens: int | None = None
    cached: bool = False


class CompletionClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        trace_id: str,
        seed: int,
    ) -> CompletionResult: ...

    def flush(self) -> None: ...


class OpenAICompletionClient:
    """Instance-scoped OpenAI-compatible client with retries and a replay cache."""

    def __init__(
        self,
        config: KQAProModelConfig,
        *,
        timeout_s: float,
        retries: int,
        cache_path: Path,
    ) -> None:
        self.config = config
        self.timeout_s = timeout_s
        self.retries = retries
        self.cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        if cache_path.exists():
            raw = read_json(cache_path)
            if not isinstance(raw, dict):
                raise ValueError(f"completion cache must be a JSON object: {cache_path}")
            self._cache = {str(key): dict(value) for key, value in raw.items()}

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        trace_id: str,
        seed: int,
    ) -> CompletionResult:
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "install aiohttp from requirements.txt for KQAPro evaluation"
            ) from exc
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": self.config.max_completion_tokens,
        }
        cache_key = stable_hash(payload)
        async with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return CompletionResult.model_validate({**cached, "cached": True})

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout_s)
                headers = {"X-Trace-ID": trace_id}
                async with (
                    aiohttp.ClientSession(timeout=timeout) as session,
                    session.post(
                        self.config.model_url.rstrip("/") + "/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as response,
                ):
                    body = await response.json()
                    if response.status != 200:
                        raise RuntimeError(f"model returned HTTP {response.status}: {body}")
                choice = body["choices"][0]["message"]
                usage = body.get("usage", {})
                result = CompletionResult(
                    content=str(choice.get("content", "")),
                    completion_tokens=(
                        int(usage["completion_tokens"])
                        if usage.get("completion_tokens") is not None
                        else None
                    ),
                )
                async with self._lock:
                    self._cache[cache_key] = result.model_dump(mode="json", exclude={"cached"})
                return result
            except (
                aiohttp.ClientError,
                TimeoutError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                last_error = exc
                if attempt < self.retries:
                    await asyncio.sleep(min(0.5 * 2**attempt, 2.0))
        raise RuntimeError(
            f"model request failed after {self.retries + 1} attempts"
        ) from last_error

    def flush(self) -> None:
        write_json(self.cache_path, self._cache)


_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL)


def _parse_direct_answer(text: str) -> AnswerSet:
    match = _ANSWER_PATTERN.search(text)
    if match is None:
        raise ValueError("missing <answer> block")
    payload = json.loads(match.group(1))
    values = payload if isinstance(payload, list) else [payload]
    if not values:
        raise ValueError("answer list is empty")
    if any(isinstance(value, dict | list) or value is None for value in values):
        raise ValueError("answers must be scalar values")
    return AnswerSet.literals([str(value) for value in values])


def _direct_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the KQAPro question directly from your internal knowledge. Do not call "
                "tools and do not produce a program. Return exactly one JSON list inside "
                "<answer>...</answer>, using entity names or literal values and no prose."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]


def _display_values(answers: AnswerSet, backend: GraphBackend) -> list[str]:
    values: list[str] = []
    for answer in answers.answers:
        rendered = str(answer.value)
        if answer.kind == "entity":
            with suppress(KeyError, ValueError):
                rendered = backend.entity_info(rendered).label
        else:
            # A direct-answer model may return an entity ID without knowing the result type.
            try:
                info = backend.entity_info(rendered)
                if info.label:
                    rendered = info.label
            except (KeyError, ValueError):
                pass
        values.append(rendered)
    return values


def _normalize_value(value: str) -> str:
    return " ".join(value.casefold().split())


def _score(predicted: AnswerSet, gold: AnswerSet, backend: GraphBackend) -> dict[str, float]:
    pred = {_normalize_value(value) for value in _display_values(predicted, backend)}
    target = {_normalize_value(value) for value in _display_values(gold, backend)}
    precision = len(pred & target) / len(pred) if pred else 0.0
    recall = len(pred & target) / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": float(pred == target),
    }


def _reason(code: str, error: Exception) -> dict[str, str]:
    if isinstance(error, GraphScriptError):
        return {"code": code, "detail": error.detail, "cause_code": error.reason_code}
    return {"code": code, "detail": str(error), "cause_code": type(error).__name__}


async def _direct_result(
    task: TaskCertificate,
    *,
    client: CompletionClient,
    backend: GraphBackend,
    model_name: str,
    seed: int,
    trace_suffix: str,
) -> dict[str, Any]:
    started = perf_counter()
    completion = await client.complete(
        _direct_prompt(task.question),
        trace_id=f"kqapro-val:{task.task_id}:{model_name}:{trace_suffix}",
        seed=seed,
    )
    predicted = _parse_direct_answer(completion.content)
    metrics = _score(predicted, task.gold_answers, backend)
    return {
        "predicted_answers": _display_values(predicted, backend),
        "raw_response": completion.content,
        "completion_tokens": completion.completion_tokens,
        "cache_hit": completion.cached,
        "latency_ms": (perf_counter() - started) * 1_000,
        **metrics,
    }


async def _evaluate_one(
    task: TaskCertificate,
    *,
    model_name: ModelStage,
    client: CompletionClient,
    backend: GraphBackend,
    relations: tuple[RelationInfo, ...],
    config: KQAProValConfig,
    seed: int,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "task_id": task.task_id,
        "source_id": task.source_id,
        "split": task.split,
        "model": model_name,
        "question": task.question,
        "gold_answers": _display_values(task.gold_answers, backend),
        "operator_tags": list(task.operator_tags),
        "fallback_used": False,
        "tool_attempted": model_name != "base",
        "tool_succeeded": False,
        "rejection_reason": None,
        "path": [],
        "support": [],
    }
    if model_name == "base":
        try:
            result = await _direct_result(
                task,
                client=client,
                backend=backend,
                model_name=model_name,
                seed=seed,
                trace_suffix="direct",
            )
            return {**common, "inference_mode": "direct", **result}
        except (TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            return {
                **common,
                "inference_mode": "direct",
                "predicted_answers": [],
                "raw_response": "",
                "completion_tokens": None,
                "cache_hit": False,
                "latency_ms": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "exact_match": 0.0,
                "rejection_reason": _reason("DIRECT_INFERENCE_FAILED", exc),
            }

    started = perf_counter()
    payload = f"Question: {task.question}"
    messages = role_prompt(
        "solver",
        payload,
        interaction_mode="graphscript",
        relation_catalog=relations,
        graphscript_version=config.graphscript_version,
    )
    primary_reason: dict[str, str] | None = None
    raw_response = ""
    completion_tokens: int | None = None
    cache_hit = False
    attempted_path: list[dict[str, Any]] = []
    try:
        completion = await client.complete(
            messages,
            trace_id=f"kqapro-val:{task.task_id}:{model_name}:graphscript",
            seed=seed,
        )
        raw_response = completion.content
        completion_tokens = completion.completion_tokens
        cache_hit = completion.cached
        try:
            script = parse_graphscript(
                completion.content, max_follow_limit=config.max_follow_limit
            )
            attempted_path = [
                op.model_dump(mode="json", by_alias=True) for op in script.ops
            ]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            primary_reason = _reason("GRAPHSCRIPT_PARSE_FAILED", exc)
            raise
        if script.version != config.graphscript_version:
            error = ValueError(
                f"expected v{config.graphscript_version}, received v{script.version}"
            )
            primary_reason = _reason("GRAPHSCRIPT_VERSION_MISMATCH", error)
            raise error
        try:
            execution = execute_graphscript(
                script,
                backend,
                allowed_relations=frozenset(relation.relation_id for relation in relations),
                max_edge_visits=config.max_edge_visits,
                max_returned_entities=config.max_returned_entities,
                trace_id=f"kqapro-val:{task.task_id}:{model_name}",
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            primary_reason = _reason("GRAPHSCRIPT_EXECUTION_FAILED", exc)
            raise
        metrics = _score(execution.answers, task.gold_answers, backend)
        return {
            **common,
            "inference_mode": "graphscript",
            "tool_succeeded": True,
            "predicted_answers": _display_values(execution.answers, backend),
            "raw_response": raw_response,
            "completion_tokens": completion_tokens,
            "cache_hit": cache_hit,
            "latency_ms": (perf_counter() - started) * 1_000,
            "path": [op.model_dump(mode="json", by_alias=True) for op in script.ops],
            "relation_path": list(execution.relation_path),
            "support": [triple.model_dump(mode="json") for triple in execution.support],
            "usage": execution.usage.model_dump(mode="json"),
            **metrics,
        }
    except (TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        primary_reason = primary_reason or _reason("MODEL_REQUEST_FAILED", exc)
        if model_name in {"sft", "grpo"}:
            try:
                fallback = await _direct_result(
                    task,
                    client=client,
                    backend=backend,
                    model_name=model_name,
                    seed=seed,
                    trace_suffix="fallback",
                )
                fallback_tokens = fallback.get("completion_tokens")
                token_values = [
                    int(value)
                    for value in (completion_tokens, fallback_tokens)
                    if value is not None
                ]
                return {
                    **common,
                    "inference_mode": "direct_fallback",
                    "fallback_used": True,
                    "rejection_reason": primary_reason,
                    "primary_raw_response": raw_response,
                    "primary_completion_tokens": completion_tokens,
                    "primary_cache_hit": cache_hit,
                    "path": attempted_path,
                    **fallback,
                    "fallback_completion_tokens": fallback_tokens,
                    "completion_tokens": sum(token_values) if token_values else None,
                    "latency_ms": (perf_counter() - started) * 1_000,
                }
            except (TypeError, ValueError, json.JSONDecodeError, RuntimeError) as fallback_error:
                return {
                    **common,
                    "inference_mode": "direct_fallback",
                    "fallback_used": True,
                    "predicted_answers": [],
                    "raw_response": "",
                    "primary_raw_response": raw_response,
                    "primary_completion_tokens": completion_tokens,
                    "primary_cache_hit": cache_hit,
                    "path": attempted_path,
                    "completion_tokens": None,
                    "cache_hit": False,
                    "latency_ms": (perf_counter() - started) * 1_000,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "exact_match": 0.0,
                    "rejection_reason": {
                        **primary_reason,
                        "fallback_code": "DIRECT_FALLBACK_FAILED",
                        "fallback_detail": str(fallback_error),
                    },
                }
        return {
            **common,
            "inference_mode": "graphscript",
            "predicted_answers": [],
            "raw_response": raw_response,
            "completion_tokens": completion_tokens,
            "cache_hit": cache_hit,
            "latency_ms": (perf_counter() - started) * 1_000,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "rejection_reason": primary_reason,
        }


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"count": 0}

    def mean(key: str) -> float:
        return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / count

    fallbacks = [row for row in rows if row["fallback_used"]]
    tool_rows = [row for row in rows if row["tool_attempted"]]
    return {
        "count": count,
        "exact_match": mean("exact_match"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "mean_latency_ms": mean("latency_ms"),
        "mean_completion_tokens": mean("completion_tokens"),
        "tool_success_rate": (
            sum(bool(row["tool_succeeded"]) for row in tool_rows) / len(tool_rows)
            if tool_rows
            else 0.0
        ),
        "fallback_rate": len(fallbacks) / count,
        "fallback_exact_match": (
            sum(float(row["exact_match"]) for row in fallbacks) / len(fallbacks)
            if fallbacks
            else 0.0
        ),
        "primary_failure_rate": (
            sum(row["rejection_reason"] is not None for row in rows) / count
        ),
        "terminal_failure_rate": (
            sum(not row["predicted_answers"] for row in rows) / count
        ),
    }


def _metrics(
    results: Sequence[dict[str, Any]], config: KQAProValConfig, model_stage: ModelStage
) -> dict[str, Any]:
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        for operator in row["operator_tags"] or ["untagged"]:
            by_operator[str(operator)].append(row)
    return {
        "dataset": "kqapro",
        "split": "val",
        "model_stage": model_stage,
        "model_id": config.model.model,
        "graph_snapshot": config.graph_snapshot,
        "graphscript_version": config.graphscript_version,
        "seed": config.seed,
        "overall": _summarize(results),
        "by_operator": {
            operator: _summarize(rows) for operator, rows in sorted(by_operator.items())
        },
    }


def _select_visual_rows(
    results: Sequence[dict[str, Any]], *, maximum: int, seed: int
) -> list[dict[str, Any]]:
    def rank(row: dict[str, Any]) -> tuple[int, str]:
        interesting = bool(row["fallback_used"] or row["rejection_reason"])
        return (0 if interesting else 1, stable_hash([seed, row["task_id"]]))

    return sorted(results, key=rank)[:maximum]


def _path_html(row: dict[str, Any]) -> str:
    if row["inference_mode"] == "direct":
        return '<p class="muted">直接回答（未访问图）</p>'
    prefix = ""
    if row["inference_mode"] == "direct_fallback":
        reason = row.get("rejection_reason") or {}
        prefix = (
            '<p class="fallback">GraphScript 失败后直接回答</p>'
            f'<p class="muted">{escape(str(reason.get("code", "unknown")))}</p>'
        )
    if not row.get("path"):
        reason = row.get("rejection_reason") or {}
        return prefix + (
            '<p class="bad">未产生可执行路径</p>'
            f'<p class="muted">{escape(str(reason.get("code", "unknown")))}</p>'
        )
    steps: list[str] = []
    for index, operation in enumerate(row["path"], start=1):
        op = escape(str(operation.get("op", "unknown")))
        detail = escape(
            json.dumps(
                {key: value for key, value in operation.items() if key != "op"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        steps.append(
            f'<li><span class="step-number">{index}</span><b>{op}</b><code>{detail}</code></li>'
        )
    support = "".join(
        "<li>"
        + " → ".join(
            escape(str(triple[key])) for key in ("subject", "relation", "object")
        )
        + "</li>"
        for triple in row.get("support", [])[:20]
    )
    support_html = (
        '<details><summary>执行证据（'
        f'{len(row.get("support", []))} 条）</summary><ul>{support}</ul></details>'
        if row.get("support")
        else ""
    )
    return prefix + f'<ol class="path">{"".join(steps)}</ol>{support_html}'


def render_kqapro_val_html(
    results: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    output_path: Path,
    *,
    maximum: int,
    seed: int,
) -> None:
    selected = _select_visual_rows(results, maximum=maximum, seed=seed)
    stage = str(metrics["model_stage"])
    summary = metrics["overall"]
    accuracy = float(summary.get("exact_match", 0.0))
    model_card = (
        f'<div class="metric"><h3>{stage.upper()}</h3><strong>{accuracy:.2%}</strong>'
        f'<div class="bar"><i style="width:{accuracy * 100:.4f}%"></i></div>'
        f'<small>F1 {float(summary.get("f1", 0.0)):.3f} · '
        f'n={summary.get("count", 0)}</small></div>'
    )
    examples: list[str] = []
    for row in selected:
        status = "correct" if row["exact_match"] else "wrong"
        answer = escape(json.dumps(row["predicted_answers"], ensure_ascii=False))
        model_result = (
            f'<section class="model {status}"><header><b>{stage.upper()}</b>'
            f'<span>{"✓" if status == "correct" else "✕"}</span></header>'
            f'<p class="answer">{answer}</p>{_path_html(row)}</section>'
        )
        examples.append(
            '<article><h2>'
            + escape(str(row["question"]))
            + '</h2><p class="gold">Gold: '
            + escape(json.dumps(row["gold_answers"], ensure_ascii=False))
            + f'</p><div class="models">{model_result}</div></article>'
        )
    css = """
:root {
  --bg:#f5f2ea; --card:#fffdf8; --ink:#1f2933; --line:#d9d3c4;
  --accent:#216869; --bad:#a23e48;
}
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,sans-serif }
main { max-width:1500px; margin:auto; padding:32px }
h1 { margin-bottom:4px }
.subtitle,.muted { color:#65717d }
.summary,.models { display:grid; grid-template-columns:minmax(0,1fr); gap:16px }
.metric,article {
  background:var(--card); border:1px solid var(--line); border-radius:14px;
  box-shadow:0 5px 20px #2d37480c;
}
.metric { padding:18px }
.metric strong { font-size:30px }
.bar { height:7px; background:#e3ded2; border-radius:9px; overflow:hidden }
.bar i { display:block; height:100%; background:var(--accent) }
article { padding:22px; margin-top:22px }
article h2 { font-size:18px }
.gold { background:#e8f1ed; padding:8px 12px; border-radius:8px }
.model {
  border:1px solid var(--line); border-top:5px solid var(--bad); padding:14px;
  border-radius:10px; min-width:0;
}
.model.correct { border-top-color:var(--accent) }
.model header { display:flex; justify-content:space-between; font-size:17px }
.answer { font-weight:650; overflow-wrap:anywhere }
.path { list-style:none; padding:0 }
.path li {
  position:relative; margin:0 0 10px 13px; padding:0 0 10px 22px;
  border-left:2px solid #b8c7c2;
}
.step-number {
  position:absolute; left:-13px; top:-2px; background:var(--accent); color:white;
  width:24px; height:24px; text-align:center; border-radius:50%;
}
code { display:block; white-space:pre-wrap; overflow-wrap:anywhere; color:#59636e; font-size:12px }
.fallback { color:#8a5b00; font-weight:650 }
.bad { color:var(--bad); font-weight:650 }
details { font-size:13px }
details ul { max-height:220px; overflow:auto; padding-left:20px }
@media(max-width:900px) {
  .summary,.models { grid-template-columns:1fr }
  main { padding:16px }
}
"""
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>KQAPro val 模型路径对比</title><style>{css}</style></head>
<body><main><h1>KQAPro val 模型路径对比</h1>
<p class="subtitle">当前阶段：{escape(stage)}；GraphScript 版本
v{escape(str(metrics['graphscript_version']))}。静态抽样 {len(selected)} 个问题。</p>
<div class="summary">{model_card}</div>{''.join(examples)}</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


async def evaluate_kqapro_val(
    input_path: Path,
    output_dir: Path,
    config: KQAProValConfig,
    *,
    model_stage: ModelStage,
    limit: int | None = None,
    selected_indices: frozenset[int] | None = None,
    backend: GraphBackend | None = None,
    client: CompletionClient | None = None,
) -> dict[str, Any]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    tasks = _load_val_tasks(
        input_path,
        config,
        limit=limit,
        selected_indices=selected_indices,
    )
    if not tasks:
        raise ValueError("KQAPro val input is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_backend = backend or backend_from_snapshot(config.graph_snapshot)
    relations = (
        load_relation_catalog(config.relation_catalog) if model_stage != "base" else ()
    )
    if model_stage != "base" and not relations:
        raise ValueError("KQAPro evaluation requires a non-empty relation catalog")
    owned_client = client is None
    effective_client = client or OpenAICompletionClient(
        config.model,
        timeout_s=config.request_timeout_s,
        retries=config.request_retries,
        cache_path=output_dir / "cache" / f"{model_stage}.json",
    )

    semaphore = asyncio.Semaphore(config.concurrency)

    async def bounded(task: TaskCertificate, index: int) -> dict[str, Any]:
        async with semaphore:
            return await _evaluate_one(
                task,
                model_name=model_stage,
                client=effective_client,
                backend=effective_backend,
                relations=relations,
                config=config,
                seed=config.seed + index,
            )

    jobs = [bounded(task, index) for index, task in enumerate(tasks)]
    try:
        results = await asyncio.gather(*jobs)
    finally:
        if owned_client:
            effective_client.flush()
    summary = _metrics(results, config, model_stage)
    summary["input"] = str(input_path)
    summary["examples"] = len(tasks)
    summary["predictions"] = str(output_dir / "predictions.parquet")
    write_records(output_dir / "predictions.parquet", results)
    write_json(output_dir / "metrics.json", summary)
    LOGGER.info("evaluated %d KQAPro val tasks for %s", len(tasks), model_stage)
    return summary


def _load_val_tasks(
    input_path: Path,
    config: KQAProValConfig,
    *,
    limit: int | None,
    selected_indices: frozenset[int] | None,
) -> list[TaskCertificate]:
    from graphtask_r1.utils import iter_records

    if selected_indices is not None and any(index < 0 for index in selected_indices):
        raise ValueError("selected indices must be non-negative")
    tasks: list[TaskCertificate] = []
    for index, raw in enumerate(iter_records(input_path)):
        if selected_indices is not None and index not in selected_indices:
            continue
        task = TaskCertificate.model_validate(raw)
        if task.split != "val":
            raise ValueError(f"KQAPro val evaluation received split {task.split!r}")
        if task.graph_snapshot != config.graph_snapshot:
            raise ValueError(
                f"task snapshot {task.graph_snapshot!r} != {config.graph_snapshot!r}"
            )
        tasks.append(task)
        if selected_indices is None and limit is not None and len(tasks) >= limit:
            break
    if selected_indices is not None and len(tasks) != len(selected_indices):
        missing = len(selected_indices) - len(tasks)
        raise IndexError(f"{missing} selected KQAPro indices are outside the dataset")
    return tasks


def inspect_kqapro_val(
    input_path: Path,
    config: KQAProValConfig,
    *,
    limit: int = 3,
    selected_indices: frozenset[int] | None = None,
    backend: GraphBackend | None = None,
) -> list[dict[str, Any]]:
    """Return a small, CLI-printable view without calling any model."""

    tasks = _load_val_tasks(
        input_path,
        config,
        limit=limit,
        selected_indices=selected_indices,
    )
    effective_backend = backend or backend_from_snapshot(config.graph_snapshot)
    return [
        {
            "task_id": task.task_id,
            "source_id": task.source_id,
            "question": task.question,
            "gold_answers": _display_values(task.gold_answers, effective_backend),
            "operator_tags": list(task.operator_tags),
        }
        for task in tasks
    ]


def compare_kqapro_val_metrics(
    metric_paths: Sequence[Path], *, output_path: Path | None = None
) -> dict[str, Any]:
    """Compare three completed single-model runs without invoking a model."""

    loaded = [read_json(path) for path in metric_paths]
    if any(not isinstance(value, dict) for value in loaded):
        raise ValueError("each KQAPro metric file must contain a JSON object")
    by_stage = {str(value.get("model_stage")): value for value in loaded}
    required = {"base", "sft", "grpo"}
    if set(by_stage) != required or len(loaded) != len(required):
        raise ValueError("comparison requires exactly one base, one sft, and one grpo metrics file")
    invariant_fields = ("dataset", "split", "graph_snapshot", "input", "examples")
    for field in invariant_fields:
        values = {str(value.get(field)) for value in loaded}
        if len(values) != 1:
            raise ValueError(f"metric files disagree on {field}: {sorted(values)}")
    stages: dict[str, dict[str, Any]] = {
        stage: {
            "model_id": str(value.get("model_id", "unknown")),
            "exact_match": float(value["overall"]["exact_match"]),
            "f1": float(value["overall"]["f1"]),
            "precision": float(value["overall"]["precision"]),
            "recall": float(value["overall"]["recall"]),
            "tool_success_rate": float(value["overall"]["tool_success_rate"]),
            "fallback_rate": float(value["overall"]["fallback_rate"]),
        }
        for stage, value in by_stage.items()
    }
    base = stages["base"]
    comparison = {
        "dataset": loaded[0]["dataset"],
        "split": loaded[0]["split"],
        "graph_snapshot": loaded[0]["graph_snapshot"],
        "input": loaded[0]["input"],
        "examples": loaded[0]["examples"],
        "stages": stages,
        "delta_vs_base": {
            stage: {
                "exact_match": values["exact_match"] - base["exact_match"],
                "f1": values["f1"] - base["f1"],
            }
            for stage, values in stages.items()
            if stage != "base"
        },
    }
    if output_path is not None:
        write_json(output_path, comparison)
    return comparison


async def visualize_kqapro_val(
    input_path: Path,
    output_dir: Path,
    config: KQAProValConfig,
    *,
    model_stage: ModelStage,
    limit: int = 3,
    selected_indices: frozenset[int] | None = None,
    backend: GraphBackend | None = None,
    client: CompletionClient | None = None,
) -> dict[str, Any]:
    """Run a bounded example evaluation and emit a standalone path report."""

    preview = inspect_kqapro_val(
        input_path,
        config,
        limit=limit,
        selected_indices=selected_indices,
        backend=backend,
    )
    metrics = await evaluate_kqapro_val(
        input_path,
        output_dir,
        config,
        model_stage=model_stage,
        limit=limit,
        selected_indices=selected_indices,
        backend=backend,
        client=client,
    )
    from graphtask_r1.utils import read_records

    results = read_records(output_dir / "predictions.parquet")
    html_path = output_dir / "paths.html"
    render_kqapro_val_html(
        results,
        metrics,
        html_path,
        maximum=len(preview),
        seed=config.seed,
    )
    console_results = [
        {
            "task_id": row["task_id"],
            "model": row["model"],
            "prediction": row["predicted_answers"],
            "correct": bool(row["exact_match"]),
            "mode": row["inference_mode"],
            "fallback_used": row["fallback_used"],
            "path": [operation.get("op") for operation in row.get("path", [])],
            "failure": row.get("rejection_reason"),
        }
        for row in results
    ]
    return {
        "dataset_preview": preview,
        "results": console_results,
        "html": str(html_path),
        "metrics": metrics["overall"],
    }
