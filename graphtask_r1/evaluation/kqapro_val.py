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
from graphtask_r1.utils import ProgressLogger, read_json, stable_hash, write_json, write_records

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
    concurrency: int = Field(default=2, ge=1)
    request_timeout_s: float = Field(default=600.0, gt=0)
    request_retries: int = Field(default=1, ge=0, le=10)
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
        error_name = type(last_error).__name__ if last_error is not None else "unknown"
        error_detail = str(last_error) if last_error is not None else "unknown error"
        raise RuntimeError(
            f"model request failed after {self.retries + 1} attempts: "
            f"{error_name}: {error_detail}"
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
    capture_execution_steps: bool,
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
        "execution_steps": [],
        "entity_details": {},
        "relation_details": {},
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
                capture_steps=capture_execution_steps,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            primary_reason = _reason("GRAPHSCRIPT_EXECUTION_FAILED", exc)
            raise
        metrics = _score(execution.answers, task.gold_answers, backend)
        serialized_steps = [step.model_dump(mode="json") for step in execution.steps]
        entity_ids = {
            entity_id
            for step in execution.steps
            for entity_id in (
                *step.selected_entities,
                *step.retrieved_entities,
                *step.discarded_entities,
                *(
                    value
                    for handle in step.input_handles.values()
                    if handle.kind == "entity"
                    for value in handle.values
                ),
                *(
                    step.output.values
                    if step.output is not None and step.output.kind == "entity"
                    else ()
                ),
                *(triple.subject for triple in step.new_evidence),
            )
        }
        attribute_ids = {
            str(value)
            for step in execution.steps
            for key in ("attribute", "relation")
            if step.operation.get("op")
            in {
                "filter_literal",
                "query_attribute",
                "query_attribute_under_condition",
                "query_attribute_qualifier",
                "select_between",
                "select_among",
            }
            and (value := step.operation.get(key)) is not None
        }
        observed_properties: dict[str, dict[str, list[str]]] = {}
        for triple in execution.support:
            if triple.relation not in attribute_ids or triple.subject not in entity_ids:
                continue
            properties = observed_properties.setdefault(triple.subject, {})
            properties.setdefault(triple.relation, []).append(triple.object)
        entity_details: dict[str, dict[str, Any]] = {}
        for entity_id in sorted(entity_ids):
            try:
                entity_details[entity_id] = backend.entity_info(entity_id).model_dump(
                    mode="json"
                )
            except (KeyError, ValueError):
                entity_details[entity_id] = {
                    "entity_id": entity_id,
                    "label": entity_id,
                    "aliases": [],
                    "type_ids": [],
                }
            entity_details[entity_id]["observed_properties"] = {
                key: sorted(set(values))
                for key, values in sorted(
                    observed_properties.get(entity_id, {}).items()
                )
            }
        used_relations = {
            triple.relation for triple in execution.support
        } | set(execution.relation_path)
        relation_details = {
            relation.relation_id: relation.model_dump(mode="json")
            for relation in relations
            if relation.relation_id in used_relations
        }
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
            "execution_steps": serialized_steps,
            "entity_details": entity_details,
            "relation_details": relation_details,
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


def _json_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _path_html(row: dict[str, Any], *, example_index: int) -> str:
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
    if row.get("execution_steps"):
        trace_payload = {
            "steps": row["execution_steps"],
            "entity_details": row.get("entity_details", {}),
            "relation_details": row.get("relation_details", {}),
        }
        return prefix + (
            f'<div class="trace-view" id="trace-{example_index}">'
            '<div class="trace-controls">'
            '<button type="button" data-action="previous">← 上一步</button>'
            '<button type="button" data-action="next">下一步 →</button>'
            '<button type="button" data-action="reset-layout">重置布局</button>'
            '<span class="trace-hint">拖动节点可调整布局</span>'
            '<span class="trace-position"></span>'
            '</div><div class="trace-layout">'
            '<nav class="trace-steps" aria-label="GraphScript steps"></nav>'
            '<section class="trace-graph-panel">'
            '<div class="trace-legend"><span class="operator">操作/流程</span>'
            '<span class="input">输入</span>'
            '<span class="retrieved">本步获取</span><span class="selected">选中/输出</span>'
            '<span class="discarded">过滤掉</span><span class="answer">数值/答案</span>'
            '<span class="deferred">延迟集合</span></div>'
            '<svg class="trace-graph" viewBox="0 0 760 430" role="img"></svg>'
            '</section><aside class="trace-inspector">'
            '<h4 class="trace-title"></h4><div class="trace-effect"></div>'
            '<div class="trace-summary"></div>'
            '<h5>操作参数</h5><pre class="trace-operation"></pre>'
            '<h5>累计预算</h5><pre class="trace-usage"></pre>'
            '<h5>节点详情</h5><pre class="trace-node-detail">点击图中的节点查看</pre>'
            '</aside></div>'
            '<details class="raw-program"><summary>查看模型生成的原始 GraphScript</summary>'
            f'<pre>{escape(str(row.get("raw_response", "")))}</pre></details>'
            '<script type="application/json" class="trace-data">'
            f'{_json_script(trace_payload)}</script>'
            '</div>'
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
    for example_index, row in enumerate(selected):
        status = "correct" if row["exact_match"] else "wrong"
        answer = escape(json.dumps(row["predicted_answers"], ensure_ascii=False))
        model_result = (
            f'<section class="model {status}"><header><b>{stage.upper()}</b>'
            f'<span>{"✓" if status == "correct" else "✕"}</span></header>'
            f'<p class="answer">{answer}</p>'
            f'{_path_html(row, example_index=example_index)}</section>'
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
.trace-view { margin-top:14px; border-top:1px solid var(--line); padding-top:14px }
.trace-controls { display:flex; align-items:center; gap:8px; margin-bottom:10px }
.trace-controls button,.trace-step {
  border:1px solid #b8c2c8; background:#fff; color:var(--ink); border-radius:8px;
  padding:7px 10px; cursor:pointer;
}
.trace-controls button:disabled { opacity:.4; cursor:default }
.trace-hint { color:#65717d; font-size:12px }
.trace-position { color:#65717d; margin-left:auto }
.trace-layout { display:grid; grid-template-columns:210px minmax(440px,1fr) 300px; gap:12px }
.trace-steps { max-height:500px; overflow:auto; padding-right:4px }
.trace-step { display:block; width:100%; text-align:left; margin-bottom:7px }
.trace-step.active {
  background:#e0f2f1; border-color:var(--accent); box-shadow:inset 4px 0 var(--accent)
}
.trace-step b { display:block }.trace-step small { display:block; color:#65717d; margin-top:2px }
.trace-graph-panel,.trace-inspector {
  border:1px solid var(--line); border-radius:10px; background:#fff
}
.trace-graph-panel { min-width:0; overflow:auto }
.trace-graph { display:block; width:100%; min-height:430px }
.trace-legend {
  display:flex; flex-wrap:wrap; gap:7px; padding:9px; border-bottom:1px solid var(--line)
}
.trace-legend span { border-radius:999px; padding:2px 7px; font-size:11px }
.trace-legend .operator { background:#f3e8ff }
.trace-legend .input { background:#dbeafe }.trace-legend .retrieved { background:#dcfce7 }
.trace-legend .selected { background:#ede9fe }.trace-legend .discarded { background:#fee2e2 }
.trace-legend .answer { background:#ffedd5 }.trace-legend .deferred { background:#fef3c7 }
.trace-inspector { padding:12px; max-height:500px; overflow:auto }
.trace-inspector h4 { margin:0 0 8px }.trace-inspector h5 { margin:13px 0 4px }
.trace-effect {
  background:#f0fdfa; border-left:4px solid var(--accent); border-radius:6px;
  margin:0 0 10px; padding:8px 10px; font-size:12px
}
.trace-inspector pre,.raw-program pre {
  white-space:pre-wrap; overflow-wrap:anywhere; font-size:11px
}
.trace-summary dl { margin:0 }.trace-summary dt { color:#65717d; font-size:11px; margin-top:7px }
.trace-summary dd { margin:1px 0; overflow-wrap:anywhere }.trace-count { font-weight:700 }
.raw-program { margin-top:10px }.trace-node,.trace-edge { cursor:pointer }
.trace-node { touch-action:none; user-select:none }.trace-node.dragging { cursor:grabbing }
@media(max-width:900px) {
  .summary,.models { grid-template-columns:1fr }
  .trace-layout { grid-template-columns:1fr }.trace-steps { max-height:190px }
  main { padding:16px }
}
"""
    javascript = r"""
const SVG_NS = "http://www.w3.org/2000/svg";
const OP_LABELS = {
  start: "起始实体", all_entities: "候选全集", resolve_entity: "解析实体",
  follow: "沿关系扩展", intersect: "取交集", union: "合并集合",
  filter_type: "按类型过滤", filter_literal: "按属性值过滤",
  filter_qualifier: "按限定符过滤", count: "计数",
  query_attribute: "查询属性", query_attribute_under_condition: "条件属性查询",
  query_attribute_qualifier: "查询属性限定符", query_relation: "查询关系",
  query_relation_qualifier: "查询关系限定符", verify: "验证条件",
  select_between: "二选一", select_among: "从集合选择",
  require_unique: "要求唯一结果", emit: "输出答案"
};
const escapeHtml = value => String(value)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const unique = values => [...new Set(values.map(String))];

function initTrace(view) {
  const data = JSON.parse(view.querySelector(".trace-data").textContent);
  const steps = data.steps || [];
  const details = data.entity_details || {};
  const relationDetails = data.relation_details || {};
  let current = 0;
  const nav = view.querySelector(".trace-steps");
  const graphPanel = view.querySelector(".trace-graph-panel");
  const svg = view.querySelector(".trace-graph");
  const position = view.querySelector(".trace-position");
  const effect = view.querySelector(".trace-effect");
  const summary = view.querySelector(".trace-summary");
  const operation = view.querySelector(".trace-operation");
  const usage = view.querySelector(".trace-usage");
  const nodeDetail = view.querySelector(".trace-node-detail");
  const previous = view.querySelector('[data-action="previous"]');
  const next = view.querySelector('[data-action="next"]');
  const resetLayout = view.querySelector('[data-action="reset-layout"]');
  const manualPositions = {};
  const MAX_GRAPH_NODES = 18;
  const label = id => details[id]?.label || relationDetails[id]?.label || String(id);
  const shortLabel = id => {
    const value = label(id);
    return value.length > 19 ? value.slice(0, 18) + "…" : value;
  };
  const handleText = handle => {
    if (!handle) return "—";
    if (handle.state === "deferred" || handle.kind === "program") {
      const limit = handle.limit ? `，最多 ${handle.limit} 个实体` : "";
      return `entity set · 延迟求值${limit}（不是空结果）`;
    }
    if (handle.state === "empty" || handle.kind === "empty") return "empty · 空集合";
    const shown = handle.values.map(value => label(value)).join(", ") || "∅";
    const count = handle.truncated
      ? `${handle.values.length} / ${handle.total_count}（截断）`
      : String(handle.total_count);
    return `${handle.kind} · ${count}: ${shown}`;
  };
  const handleCount = handle => {
    if (!handle) return "无输出";
    if (handle.state === "deferred" || handle.kind === "program") {
      return handle.limit ? `延迟集合 ≤ ${handle.limit}` : "延迟集合";
    }
    return handle.truncated
      ? `${handle.values.length} / ${handle.total_count}` : String(handle.total_count);
  };
  const inputValuesFor = step => Object.values(step.input_handles || {})
    .flatMap(handle => handle.values || []);
  const stepValues = step => unique([
    ...inputValuesFor(step), ...(step.output?.values || []),
    ...(step.retrieved_entities || []), ...(step.selected_entities || []),
    ...(step.discarded_entities || []),
    ...(step.new_evidence || []).flatMap(edge => [edge.subject, edge.object])
  ]);
  const operationDetail = operation => {
    const op = operation.op;
    if (op === "filter_type") return `type = ${operation.type_id}`;
    if (op === "filter_literal") {
      return `${operation.relation} ${operation.comparator} ${operation.value?.value}`;
    }
    if (op === "filter_qualifier") {
      return `${operation.qualifier} ${operation.comparator} ${operation.value?.value}`;
    }
    if (op === "follow") return `${operation.direction} · ${operation.relation}`;
    if (op.startsWith("query_")) {
      return operation.attribute || operation.relation || operation.qualifier || "query";
    }
    if (op.startsWith("select_")) return `${operation.attribute} · ${operation.mode}`;
    if (op === "verify") return `${operation.comparator} ${operation.value?.value}`;
    if (op === "all_entities") return `limit = ${operation.max_results}`;
    return "";
  };
  const firstSeen = {};
  steps.forEach((step, index) => stepValues(step).forEach(value => {
    if (firstSeen[value] === undefined) firstSeen[value] = index;
  }));
  const representatives = steps.flatMap(step => unique([
    ...(step.selected_entities || []), ...(step.output?.values || []),
    ...(step.retrieved_entities || []), ...inputValuesFor(step),
    ...(step.new_evidence || []).flatMap(edge => [edge.subject, edge.object])
  ]).slice(0, 1));
  const finalValues = steps.length ? steps.at(-1).output?.values || [] : [];
  const allCandidates = unique(steps.flatMap(stepValues));
  const initialValues = steps[0] ? stepValues(steps[0]) : [];
  const stableCandidates = unique([
    ...initialValues.slice(0, 2), ...finalValues.slice(0, 3),
    ...representatives, ...initialValues, ...finalValues, ...allCandidates
  ]);
  const stableNodeIds = stableCandidates.slice(0, MAX_GRAPH_NODES);
  const producerByHandle = {};
  steps.forEach((step, index) => {
    if (step.output_handle) producerByHandle[step.output_handle] = index;
  });
  const virtualMeta = {};
  steps.forEach((step, index) => {
    const operatorId = `__operator__${index}`;
    virtualMeta[operatorId] = {
      kind: "operator",
      step: index,
      operation: step.operation,
      label: OP_LABELS[step.operation.op] || step.operation.op,
      detail: operationDetail(step.operation)
    };
    firstSeen[operatorId] = index;
    if (step.output?.state !== "deferred") return;
    const id = `__deferred__${index}`;
    virtualMeta[id] = {
      kind: "deferred",
      step: index,
      handle: step.output_handle || `step-${index + 1}`,
      limit: step.output.limit,
      operation: step.operation
    };
    firstSeen[id] = index;
  });
  const plannedNodeIds = [...Object.keys(virtualMeta), ...stableNodeIds];
  const layoutGroups = new Map();
  plannedNodeIds.forEach(id => {
    const column = firstSeen[id] ?? 0;
    if (!layoutGroups.has(column)) layoutGroups.set(column, []);
    layoutGroups.get(column).push(id);
  });
  const layoutColumns = [];
  [...layoutGroups.keys()].sort((left, right) => left - right).forEach(column => {
    const values = layoutGroups.get(column);
    for (let offset = 0; offset < values.length; offset += 6) {
      layoutColumns.push(values.slice(offset, offset + 6));
    }
  });
  const graphWidth = Math.max(760, 170 * layoutColumns.length + 20);
  const basePositions = {};
  layoutColumns.forEach((values, columnIndex) => {
    const x = layoutColumns.length === 1 ? graphWidth / 2 : 85 + columnIndex * 170;
    values.forEach((id, rowIndex) => {
      basePositions[id] = {x, y: 45 + (rowIndex + 1) * (335 / (values.length + 1))};
    });
  });

  function effectText(step) {
    const op = step.operation.op || "unknown";
    const inputs = Object.values(step.input_handles || {});
    const inputText = inputs.length ? inputs.map(handleCount).join(" + ") : "无输入";
    const outputText = handleCount(step.output);
    const selectedText = (step.selected_entities || []).length
      ? (step.selected_entities || []).map(label).join(", ") : "无实体节点";
    if (op === "all_entities") {
      return `建立候选全集查询，上限 ${step.operation.max_results}；本步只登记查询，` +
        "由后续 filter/query 在图后端物化，因此不是 0 个结果。";
    }
    if (op === "resolve_entity") {
      return `将“${step.operation.query}”解析为 ${outputText}：${selectedText}。`;
    }
    if (op === "follow") {
      return `从 ${inputText} 沿 ${step.operation.direction} 方向的 ` +
        `${step.operation.relation} 扩展，得到 ${outputText}，新增 ` +
        `${step.new_evidence_total} 条关系证据。`;
    }
    if (["filter_type", "filter_literal", "filter_qualifier"].includes(op)) {
      const deferred = inputs.some(handle => handle.state === "deferred")
        ? "输入是延迟集合，过滤与物化在同一次后端查询中完成；" : "";
      return `${deferred}从 ${inputText} 中保留 ${outputText}：${selectedText}。`;
    }
    if (["intersect", "union"].includes(op)) {
      return `${op === "intersect" ? "求交集" : "合并集合"}：${inputText} → ${outputText}。`;
    }
    if (op === "count") return `统计 ${inputText}，得到计数 ${outputText}。`;
    if (op.startsWith("query_")) {
      return `对 ${inputText} 执行查询，得到 ${outputText}：` +
        `${(step.output?.values || []).map(label).join(", ") || "无可展示值"}。`;
    }
    if (op.startsWith("select_")) {
      return `按 ${step.operation.attribute} 执行 ${step.operation.mode} 选择，` +
        `保留 ${outputText}：${selectedText}。`;
    }
    if (op === "verify") return `验证 ${inputText}，得到 ${outputText}。`;
    if (op === "require_unique") return `确认 ${inputText} 中恰好只有一个结果。`;
    if (op === "emit") return `输出最终答案 ${outputText}：${selectedText}。`;
    return `${inputText} → ${outputText}。`;
  }

  steps.forEach((step, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "trace-step";
    const op = step.operation.op || "unknown";
    const inputs = Object.keys(step.input_handles || {}).join(", ") || "root";
    const output = step.output_handle || (op === "emit" ? "answer" : "—");
    const title = document.createElement("b");
    title.textContent = `${index + 1}. ${op} · ${OP_LABELS[op] || "执行操作"}`;
    const flow = document.createElement("small");
    flow.textContent = `${inputs} → ${output} · ${handleCount(step.output)}`;
    button.append(title, flow);
    button.addEventListener("click", () => { current = index; render(); });
    nav.appendChild(button);
  });

  function summaryHtml(step) {
    const inputs = Object.entries(step.input_handles || {}).map(([name, handle]) =>
      `<dt>输入 ${escapeHtml(name)}</dt><dd>${escapeHtml(handleText(handle))}</dd>`
    ).join("");
    const list = values => values?.length
      ? values.map(value => escapeHtml(label(value))).join(", ") : "—";
    const evidence = step.new_evidence_total
      ? `${step.new_evidence.length} / ${step.new_evidence_total}` +
        (step.evidence_truncated ? "（截断）" : "") : "0";
    return `<dl>${inputs}` +
      `<dt>输出 ${escapeHtml(step.output_handle || "answer")}</dt>` +
      `<dd>${escapeHtml(handleText(step.output))}</dd>` +
      `<dt>本步获取的节点</dt><dd>${list(step.retrieved_entities)}</dd>` +
      `<dt>本步选中/保留</dt><dd>${list(step.selected_entities)}</dd>` +
      `<dt>本步过滤掉</dt><dd>${list(step.discarded_entities)}</dd>` +
      `<dt>新增证据</dt><dd class="trace-count">${escapeHtml(evidence)}</dd>` +
      `<dt>步骤耗时</dt><dd>${Number(step.latency_ms).toFixed(2)} ms</dd></dl>`;
  }

  function renderGraph(stepIndex) {
    svg.replaceChildren();
    const currentStep = steps[stepIndex];
    const visibleSteps = steps.slice(0, stepIndex + 1);
    const inputValues = Object.values(currentStep.input_handles || {})
      .flatMap(handle => handle.values || []);
    const outputValues = currentStep.output?.values || [];
    const selected = new Set((currentStep.selected_entities || []).map(String));
    const retrieved = new Set((currentStep.retrieved_entities || []).map(String));
    const discarded = new Set((currentStep.discarded_entities || []).map(String));
    const inputs = new Set(inputValues.map(String));
    const answerValues = new Set(
      currentStep.output?.kind === "answer" ? outputValues.map(String) : []
    );
    const cumulativeEdges = visibleSteps.flatMap(step => step.new_evidence || []);
    const currentEdges = new Set((currentStep.new_evidence || []).map(edge =>
      `${edge.subject}\u0000${edge.relation}\u0000${edge.object}`
    ));
    const nodeIds = plannedNodeIds.filter(id => (firstSeen[id] ?? 0) <= stepIndex);
    const allowed = new Set(nodeIds);
    svg.setAttribute("viewBox", `0 0 ${graphWidth} 430`);
    svg.style.minWidth = `${graphWidth}px`;
    const positions = {};
    nodeIds.forEach(id => {
      positions[id] = manualPositions[id] || basePositions[id];
    });

    const edges = cumulativeEdges.filter(edge =>
      allowed.has(String(edge.subject)) && allowed.has(String(edge.object))
    ).map(edge => ({...edge, type: "knowledge"}));
    visibleSteps.forEach((step, visibleIndex) => {
      const operator = `__operator__${visibleIndex}`;
      Object.keys(step.input_handles || {}).forEach(handleName => {
        const producer = producerByHandle[handleName];
        if (producer === undefined) return;
        edges.push({
          subject: `__operator__${producer}`,
          relation: handleName,
          object: operator,
          type: "process",
          current: visibleIndex === stepIndex
        });
      });
      const resultNodes = unique([
        ...(step.selected_entities || []), ...(step.output?.values || []),
        ...(step.retrieved_entities || [])
      ]).filter(value => allowed.has(value)).slice(0, 3);
      resultNodes.forEach(target => edges.push({
        subject: operator,
        relation: step.output_handle || (step.operation.op === "emit" ? "answer" : "result"),
        object: target,
        type: "process-result",
        current: visibleIndex === stepIndex
      }));
      if (step.output?.state === "deferred") {
        edges.push({
          subject: operator,
          relation: step.output_handle || "deferred",
          object: `__deferred__${visibleIndex}`,
          type: "process-result",
          current: visibleIndex === stepIndex
        });
      }
      Object.entries(step.input_handles || {}).forEach(([handleName, handle]) => {
        if (handle.state !== "deferred") return;
        const producer = producerByHandle[handleName];
        const source = `__deferred__${producer}`;
        if (!allowed.has(source)) return;
        edges.push({
          subject: source,
          relation: handleName,
          object: operator,
          type: "dataflow",
          current: visibleIndex === stepIndex
        });
      });
    });

    const defs = document.createElementNS(SVG_NS, "defs");
    const marker = document.createElementNS(SVG_NS, "marker");
    marker.setAttribute("id", `${view.id}-arrow`);
    marker.setAttribute("viewBox", "0 0 10 10"); marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "5"); marker.setAttribute("markerWidth", "6");
    marker.setAttribute("markerHeight", "6"); marker.setAttribute("orient", "auto-start-reverse");
    const arrow = document.createElementNS(SVG_NS, "path");
    arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z"); arrow.setAttribute("fill", "#718096");
    marker.appendChild(arrow); defs.appendChild(marker); svg.appendChild(defs);

    const edgeElements = [];
    const updateEdge = item => {
      const source = positions[item.edge.subject];
      const target = positions[item.edge.object];
      if (!source || !target) return;
      const dx = target.x - source.x, dy = target.y - source.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const sourceOffset = virtualMeta[item.edge.subject]?.kind === "operator" ? 82 : 66;
      const targetOffset = virtualMeta[item.edge.object]?.kind === "operator" ? 82 : 66;
      item.line.setAttribute("x1", source.x + dx * sourceOffset / distance);
      item.line.setAttribute("y1", source.y + dy * 24 / distance);
      item.line.setAttribute("x2", target.x - dx * targetOffset / distance);
      item.line.setAttribute("y2", target.y - dy * 24 / distance);
      item.label.setAttribute("x", (source.x + target.x) / 2);
      item.label.setAttribute("y", (source.y + target.y) / 2 - 5);
    };
    edges.forEach(edge => {
      const source = positions[edge.subject], target = positions[edge.object];
      if (!source || !target) return;
      const key = `${edge.subject}\u0000${edge.relation}\u0000${edge.object}`;
      const line = document.createElementNS(SVG_NS, "line");
      line.classList.add("trace-edge");
      const isCurrent = edge.current || currentEdges.has(key);
      const historicalColor = edge.type === "process" || edge.type === "process-result"
        ? "#7c3aed" : edge.type === "dataflow" ? "#b45309" : "#94a3b8";
      line.setAttribute("stroke", isCurrent ? "#16a34a" : historicalColor);
      line.setAttribute("stroke-width", isCurrent ? "3" : "1.5");
      if (edge.type === "dataflow") line.setAttribute("stroke-dasharray", "6 4");
      line.setAttribute("marker-end", `url(#${view.id}-arrow)`); svg.appendChild(line);
      const edgeLabel = document.createElementNS(SVG_NS, "text");
      edgeLabel.setAttribute("text-anchor", "middle"); edgeLabel.setAttribute("font-size", "10");
      edgeLabel.classList.add("trace-edge");
      edgeLabel.setAttribute("fill", "#475569");
      edgeLabel.textContent = edge.type !== "knowledge"
        ? edge.relation
        : relationDetails[edge.relation]?.label || edge.relation;
      const showRelation = () => {
        if (edge.type !== "knowledge") {
          nodeDetail.textContent = JSON.stringify({
            role: edge.type, handle: edge.relation,
            from: edge.subject, to: edge.object
          }, null, 2);
          return;
        }
        const relation = relationDetails[edge.relation] || {label: edge.relation};
        nodeDetail.textContent = JSON.stringify({
          id: edge.relation, role: "relation", ...relation,
          edge: {subject: edge.subject, object: edge.object}
        }, null, 2);
      };
      line.addEventListener("click", showRelation);
      edgeLabel.addEventListener("click", showRelation);
      svg.appendChild(edgeLabel);
      const item = {edge, line, label: edgeLabel};
      edgeElements.push(item);
      updateEdge(item);
    });

    nodeIds.forEach(id => {
      const point = positions[id];
      const group = document.createElementNS(SVG_NS, "g");
      group.classList.add("trace-node");
      group.setAttribute("transform", `translate(${point.x},${point.y})`);
      let fill = details[id] ? "#f1f5f9" : "#ffedd5", stroke = "#64748b", role = "context";
      const virtual = virtualMeta[id];
      const isOperator = virtual?.kind === "operator";
      const isDeferred = virtual?.kind === "deferred";
      if (isOperator) { fill = "#f3e8ff"; stroke = "#7c3aed"; role = "operator"; }
      if (isDeferred) { fill = "#fef3c7"; stroke = "#b45309"; role = "deferred-set"; }
      if (inputs.has(id)) { fill = "#dbeafe"; stroke = "#2563eb"; role = "input"; }
      if (retrieved.has(id)) { fill = "#dcfce7"; stroke = "#16a34a"; role = "retrieved"; }
      if (selected.has(id)) { fill = "#ede9fe"; stroke = "#7c3aed"; role = "selected"; }
      if (discarded.has(id)) { fill = "#fee2e2"; stroke = "#dc2626"; role = "discarded"; }
      if (answerValues.has(id)) { fill = "#ffedd5"; stroke = "#ea580c"; role = "answer"; }
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", isOperator ? "-82" : "-66"); rect.setAttribute("y", "-23");
      rect.setAttribute("width", isOperator ? "164" : "132");
      rect.setAttribute("height", "46");
      rect.setAttribute("rx", details[id] ? "23" : isOperator ? "6" : "12");
      rect.setAttribute("fill", fill);
      rect.setAttribute("stroke", stroke); rect.setAttribute("stroke-width", "2");
      group.appendChild(rect);
      const title = document.createElementNS(SVG_NS, "text");
      title.setAttribute("text-anchor", "middle"); title.setAttribute("y", "-2");
      title.setAttribute("font-size", "12"); title.setAttribute("font-weight", "700");
      title.textContent = isOperator ? `${virtual.step + 1}. ${virtual.operation.op}`
        : isDeferred ? `All entities${virtual.limit ? ` ≤${virtual.limit}` : ""}`
        : shortLabel(id);
      group.appendChild(title);
      const identifier = document.createElementNS(SVG_NS, "text");
      identifier.setAttribute("text-anchor", "middle"); identifier.setAttribute("y", "13");
      identifier.setAttribute("font-size", "8"); identifier.setAttribute("fill", "#64748b");
      const typeText = details[id]?.type_ids?.length
        ? ` · ${details[id].type_ids.slice(0, 2).join("/")}` : "";
      identifier.textContent = isOperator
        ? `${virtual.label}${virtual.detail ? ` · ${virtual.detail}` : ""}`
        : isDeferred ? `${virtual.handle} · 延迟求值`
        : `${String(id).length > 17 ? String(id).slice(0, 16) + "…" : String(id)}${typeText}`;
      group.appendChild(identifier);
      let dragStart = null;
      let dragged = false;
      group.addEventListener("pointerdown", event => {
        event.preventDefault();
        dragged = false;
        dragStart = {x: event.clientX, y: event.clientY};
        group.classList.add("dragging");
        group.setPointerCapture(event.pointerId);
      });
      group.addEventListener("pointermove", event => {
        if (!dragStart || !group.hasPointerCapture(event.pointerId)) return;
        const matrix = svg.getScreenCTM();
        if (!matrix) return;
        const cursor = svg.createSVGPoint();
        cursor.x = event.clientX; cursor.y = event.clientY;
        const transformed = cursor.matrixTransform(matrix.inverse());
        const nextPoint = {
          x: Math.max(70, Math.min(graphWidth - 70, transformed.x)),
          y: Math.max(28, Math.min(400, transformed.y))
        };
        dragged = dragged || Math.hypot(
          event.clientX - dragStart.x, event.clientY - dragStart.y
        ) > 3;
        positions[id] = nextPoint;
        manualPositions[id] = nextPoint;
        group.setAttribute("transform", `translate(${nextPoint.x},${nextPoint.y})`);
        edgeElements.forEach(updateEdge);
      });
      group.addEventListener("pointerup", event => {
        dragStart = null;
        group.classList.remove("dragging");
        if (group.hasPointerCapture(event.pointerId)) {
          group.releasePointerCapture(event.pointerId);
        }
      });
      group.addEventListener("pointercancel", () => {
        dragStart = null;
        group.classList.remove("dragging");
      });
      group.addEventListener("click", () => {
        if (dragged) { dragged = false; return; }
        const detail = isDeferred
          ? {id, role, ...virtual, note: "延迟集合不是空集合"}
          : isOperator ? {id, role, ...virtual}
          : {id, role, ...(details[id] || {value: id})};
        nodeDetail.textContent = JSON.stringify(detail, null, 2);
      });
      svg.appendChild(group);
    });

    if (allCandidates.length > MAX_GRAPH_NODES) {
      const note = document.createElementNS(SVG_NS, "text");
      note.setAttribute("x", String(graphWidth - 20)); note.setAttribute("y", "418");
      note.setAttribute("text-anchor", "end"); note.setAttribute("font-size", "10");
      note.setAttribute("fill", "#b45309");
      note.textContent = `固定显示 ${MAX_GRAPH_NODES} / ${allCandidates.length} 个关键节点；` +
        "已出现节点不会在后续步骤消失";
      svg.appendChild(note);
    }
    const currentIds = unique([
      `__operator__${stepIndex}`,
      ...selected, ...outputValues, ...retrieved, ...inputValues,
      `__deferred__${stepIndex}`
    ]);
    const focusId = currentIds.find(id => allowed.has(id));
    return focusId ? positions[focusId]?.x : undefined;
  }

  function render() {
    const step = steps[current];
    if (!step) return;
    nav.querySelectorAll(".trace-step").forEach((button, index) =>
      button.classList.toggle("active", index === current)
    );
    position.textContent = `${current + 1} / ${steps.length}`;
    previous.disabled = current === 0; next.disabled = current === steps.length - 1;
    const opLabel = OP_LABELS[step.operation.op] || "执行操作";
    view.querySelector(".trace-title").textContent =
      `${current + 1}. ${step.operation.op} · ${opLabel}`;
    effect.textContent = effectText(step);
    summary.innerHTML = summaryHtml(step);
    operation.textContent = JSON.stringify(step.operation, null, 2);
    usage.textContent = JSON.stringify(step.cumulative_usage, null, 2);
    nodeDetail.textContent = "点击图中的节点查看 label、ID、alias、type 与本步角色";
    const focusX = renderGraph(current);
    if (focusX !== undefined) {
      requestAnimationFrame(() => {
        graphPanel.scrollLeft = Math.max(0, focusX - graphPanel.clientWidth / 2);
      });
    }
  }
  previous.addEventListener("click", () => { if (current > 0) { current--; render(); } });
  next.addEventListener("click", () => {
    if (current < steps.length - 1) { current++; render(); }
  });
  resetLayout.addEventListener("click", () => {
    Object.keys(manualPositions).forEach(id => delete manualPositions[id]);
    render();
  });
  render();
}
document.querySelectorAll(".trace-view").forEach(initTrace);
"""
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>KQAPro val 单模型执行路径</title><style>{css}</style></head>
<body><main><h1>KQAPro val 单模型执行路径</h1>
<p class="subtitle">当前阶段：{escape(stage)}；GraphScript 版本
v{escape(str(metrics['graphscript_version']))}。静态抽样 {len(selected)} 个问题。</p>
<div class="summary">{model_card}</div>{''.join(examples)}</main>
<script>{javascript}</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _progress_bar(completed: int, total: int, *, width: int = 20) -> str:
    filled = width if total <= 0 else min(width, completed * width // total)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


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
    capture_execution_steps: bool = False,
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

    async def bounded(task: TaskCertificate, index: int) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            return (
                index,
                await _evaluate_one(
                    task,
                    model_name=model_stage,
                    client=effective_client,
                    backend=effective_backend,
                    relations=relations,
                    config=config,
                    seed=config.seed + index,
                    capture_execution_steps=capture_execution_steps,
                ),
            )

    pending = {
        asyncio.create_task(bounded(task, index)) for index, task in enumerate(tasks)
    }
    ordered_results: list[dict[str, Any] | None] = [None] * len(tasks)
    completed = 0
    correct = 0
    tool_successes = 0
    fallbacks = 0
    terminal_failures = 0
    cache_hits = 0
    progress = ProgressLogger(
        "evaluate.kqapro_val",
        total=len(tasks),
        interval_s=5.0,
    )
    progress.start(
        model_stage=model_stage,
        concurrency=config.concurrency,
        request_timeout_s=config.request_timeout_s,
        request_retries=config.request_retries,
        bar=_progress_bar(0, len(tasks)),
    )
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                timeout=5.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for future in done:
                index, row = future.result()
                ordered_results[index] = row
                completed += 1
                correct += int(bool(row["exact_match"]))
                tool_successes += int(bool(row["tool_succeeded"]))
                fallbacks += int(bool(row["fallback_used"]))
                terminal_failures += int(not row["predicted_answers"])
                cache_hits += int(bool(row["cache_hit"]))
            progress.update(
                completed,
                pending=len(pending),
                correct=correct,
                tool_successes=tool_successes,
                fallbacks=fallbacks,
                terminal_failures=terminal_failures,
                cache_hits=cache_hits,
                bar=_progress_bar(completed, len(tasks)),
            )
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if owned_client:
            effective_client.flush()
    results = [row for row in ordered_results if row is not None]
    if len(results) != len(tasks):
        raise RuntimeError(
            f"KQAPro evaluation completed {len(results)} of {len(tasks)} tasks"
        )
    progress.finish(
        completed,
        correct=correct,
        tool_successes=tool_successes,
        fallbacks=fallbacks,
        terminal_failures=terminal_failures,
        cache_hits=cache_hits,
        bar=_progress_bar(completed, len(tasks)),
    )
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
        capture_execution_steps=True,
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
            "execution_steps": row.get("execution_steps", []),
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
