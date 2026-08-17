"""ms-swift v3.6.4 runtime plugin for existing GraphTask Parquet files.

This module is imported by ``swift`` through ``--external_plugins``. Importing the rest of
GraphTask does not require ms-swift to be installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from graphtask_r1.envs.graph_query import execute_compact_query
from graphtask_r1.envs.text_search import execute_text_search
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.training.json_compat import to_json_compatible
from graphtask_r1.training.ms_swift_data import convert_rl_row, convert_sft_row
from graphtask_r1.training.ms_swift_reward import compute_score

try:
    from swift.llm.dataset import DatasetMeta, RowPreprocessor, register_dataset
    from swift.plugin import ORM, orms
except ImportError as exc:  # pragma: no cover - exercised on the training server
    raise ImportError(
        "Install the pinned ms-swift environment before loading the GraphTask plugin"
    ) from exc


logger = logging.getLogger(__name__)


class GraphTaskSFTPreprocessor(RowPreprocessor):  # type: ignore[misc]
    def preprocess(self, row: dict[str, Any]) -> dict[str, object]:
        return convert_sft_row(row)


class GraphTaskRLPreprocessor(RowPreprocessor):  # type: ignore[misc]
    def preprocess(self, row: dict[str, Any]) -> dict[str, object]:
        return convert_rl_row(row)


def _register_data() -> None:
    kind = os.environ.get("GRAPHTASK_MS_SWIFT_DATA_KIND", "")
    if not kind:
        return
    if kind not in {"sft", "rl"}:
        raise ValueError("GRAPHTASK_MS_SWIFT_DATA_KIND must be 'sft' or 'rl'")
    train_path = os.environ.get("GRAPHTASK_MS_SWIFT_TRAIN_DATA", "")
    val_path = os.environ.get("GRAPHTASK_MS_SWIFT_VAL_DATA", train_path)
    if not train_path:
        raise ValueError("GRAPHTASK_MS_SWIFT_TRAIN_DATA is required")
    for path in {train_path, val_path}:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    preprocessor = GraphTaskSFTPreprocessor() if kind == "sft" else GraphTaskRLPreprocessor()
    register_dataset(
        DatasetMeta(
            dataset_name="graphtask-train",
            dataset_path=train_path,
            preprocess_func=preprocessor,
        )
    )
    register_dataset(
        DatasetMeta(
            dataset_name="graphtask-val",
            dataset_path=val_path,
            preprocess_func=preprocessor,
        )
    )


def _batch(value: object, size: int, *, default: object) -> list[object]:
    if isinstance(value, list):
        if len(value) != size:
            raise ValueError(f"reward column has {len(value)} rows; expected {size}")
        return value
    if value is None:
        value = default
    return [value for _ in range(size)]


class GraphTaskReward(ORM):  # type: ignore[misc]
    """Return the total reward and emit all auditable components as structured logs."""

    def __init__(self) -> None:
        super().__init__()
        self._metrics_sequence = 0
        metrics_dir = os.environ.get("GRAPHTASK_REWARD_METRICS_DIR")
        rank = os.environ.get("RANK", "0")
        safe_rank = rank if rank.isdigit() else "unknown"
        self._metrics_path = (
            Path(metrics_dir) / f"reward_components.rank-{safe_rank}.jsonl"
            if metrics_dir
            else None
        )

    def _record_metrics(self, event: dict[str, object]) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def __call__(
        self,
        completions: list[str],
        data_source: object = None,
        ground_truth: object = None,
        extra_info: object = None,
        **kwargs: object,
    ) -> list[float]:
        del kwargs
        size = len(completions)
        sources = _batch(data_source, size, default="graphtask/solver")
        truths = _batch(ground_truth, size, default="")
        infos = _batch(extra_info, size, default={})

        normalized_infos: list[dict[str, Any]] = []
        for raw_info in infos:
            info = to_json_compatible(raw_info)
            if not isinstance(info, dict):
                raise ValueError("extra_info reward column must contain objects")
            normalized_infos.append(info)

        async def score_one(index: int) -> dict[str, float]:
            return await compute_score(
                str(sources[index]),
                completions[index],
                str(truths[index]),
                normalized_infos[index],
            )

        async def score_all() -> list[dict[str, float]]:
            return list(await asyncio.gather(*(score_one(index) for index in range(size))))

        results = asyncio.run(score_all())
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        role_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        role_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        role_samples: dict[str, int] = defaultdict(int)
        for index, result in enumerate(results):
            metrics = dict(result)
            role_weight = float(normalized_infos[index].get("role_weight", 1.0))
            if role_weight:
                metrics["unweighted_score"] = float(result["score"]) / role_weight
            source = str(sources[index])
            role = source.rsplit("/", maxsplit=1)[-1]
            role_samples[role] += 1
            for name, value in metrics.items():
                sums[name] += float(value)
                counts[name] += 1
                role_sums[role][name] += float(value)
                role_counts[role][name] += 1
        components = {name: sums[name] / counts[name] for name in sorted(sums) if counts[name]}
        roles = {
            role: {
                "samples": role_samples[role],
                "means": {
                    name: role_sums[role][name] / count
                    for name, count in sorted(role_counts[role].items())
                    if count
                },
            }
            for role in sorted(role_samples)
        }
        self._metrics_sequence += 1
        event: dict[str, object] = {
            "event": "graphtask_reward_components",
            "sequence": self._metrics_sequence,
            "rank": os.environ.get("RANK", "0"),
            "world_size": os.environ.get("WORLD_SIZE", "1"),
            "samples": size,
            "means": components,
            "roles": roles,
        }
        self._record_metrics(event)
        logger.info(json.dumps(event, ensure_ascii=False, sort_keys=True))
        return [float(result["score"]) for result in results]


def _tool_calls(response_choice: Any) -> list[Any]:
    calls = getattr(getattr(response_choice, "message", None), "tool_calls", None)
    return list(calls or [])


def _parse_arguments(value: object) -> dict[str, object]:
    normalized = to_json_compatible(value)
    if isinstance(normalized, str):
        normalized = json.loads(normalized)
    if not isinstance(normalized, dict):
        raise ValueError("tool arguments must be a JSON object")
    return normalized


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid integer limit")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"expected an integer limit, got {type(value).__name__}")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError("expected a list")
    return [str(item) for item in value]


class GraphTaskSolverScheduler:
    """Instance-scoped Hermes tool scheduler for Solver-only graph and passage rollout."""

    def __init__(self, *args: object, max_turns: int | None = None, **kwargs: object) -> None:
        del args, kwargs
        self.max_turns = max_turns
        self._backends: dict[str, GraphBackend] = {}

    def _backend(self, snapshot: str) -> GraphBackend:
        if snapshot not in self._backends:
            self._backends[snapshot] = backend_from_snapshot(snapshot)
        return self._backends[snapshot]

    @staticmethod
    def _info(infer_request: Any) -> dict[str, object]:
        data_dict = to_json_compatible(getattr(infer_request, "data_dict", {}))
        if not isinstance(data_dict, dict):
            raise ValueError("rollout data_dict must be an object")
        info = data_dict.get("extra_info", {})
        if not isinstance(info, dict):
            raise ValueError("rollout extra_info must be an object")
        return info

    @staticmethod
    def _state(infer_request: Any, info: Mapping[str, object]) -> dict[str, object]:
        data_dict = infer_request.data_dict
        state = data_dict.get("_graphtask_session")
        if state is None:
            state = {
                "calls": 0,
                "invalid_calls": 0,
                "edge_visits": 0,
                "visible_entities": _string_list(info.get("topic_entity_ids", [])),
            }
            data_dict["_graphtask_session"] = state
        if not isinstance(state, dict):
            raise ValueError("invalid GraphTask rollout session state")
        return state

    def check_finished(self, infer_request: Any, response_choice: Any, current_turn: int) -> bool:
        del infer_request
        if getattr(response_choice, "finish_reason", None) == "length":
            return True
        if self.max_turns is not None and current_turn >= self.max_turns:
            return True
        return not _tool_calls(response_choice)

    def _graph_search(
        self,
        parameters: Mapping[str, object],
        info: Mapping[str, object],
        state: dict[str, object],
    ) -> str:
        snapshot = str(info.get("graph_snapshot", "kqapro-v1"))
        if "query" in parameters:
            max_entities = min(
                512,
                max(1, _int_value(info.get("max_returned_entities", 512))),
            )
            result = execute_compact_query(
                self._backend(snapshot),
                parameters["query"],
                max_limit=max_entities,
            )
            visits = max(1, len(result.entities), len(result.values))
            state["edge_visits"] = _int_value(state.get("edge_visits", 0)) + visits
            visible = set(_string_list(state.get("visible_entities", [])))
            visible.update(entity.entity_id for entity in result.entities)
            state["visible_entities"] = sorted(visible)
            return result.model_dump_json()

        raw_entities = parameters.get("entity_ids")
        if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, str | bytes):
            raise ValueError("entity_ids must be a non-empty list")
        entity_ids = _string_list(raw_entities)
        if not entity_ids:
            raise ValueError("entity_ids must be a non-empty list")
        raw_relations = parameters.get("relation_ids", [])
        if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, str | bytes):
            raise ValueError("relation_ids must be a list")
        relation_ids = _string_list(raw_relations)
        max_edges = _int_value(info.get("max_edge_visits", 200))
        remaining = max_edges - _int_value(state.get("edge_visits", 0))
        if remaining <= 0:
            raise ValueError("graph-search edge budget exhausted")
        limit = min(max(1, _int_value(parameters.get("limit", 50))), 100, remaining)
        triples = self._backend(snapshot).neighbors(
            entity_ids,
            direction=str(parameters.get("direction", "both")),
            relation_ids=relation_ids or None,
            limit=limit,
            trace_id=f"{info.get('task_id', 'solver')}:{state.get('calls', 0)}",
        )
        state["edge_visits"] = _int_value(state.get("edge_visits", 0)) + len(triples)
        visible = set(_string_list(state.get("visible_entities", [])))
        visible.update(value for triple in triples for value in (triple.subject, triple.object))
        state["visible_entities"] = sorted(visible)
        return json.dumps(
            [triple.model_dump(mode="json") for triple in triples],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _execute_tool(
        self,
        name: str,
        parameters: Mapping[str, object],
        info: Mapping[str, object],
        state: dict[str, object],
    ) -> str:
        state["calls"] = _int_value(state.get("calls", 0)) + 1
        max_calls = _int_value(info.get("max_tool_calls", self.max_turns or 8))
        if _int_value(state["calls"]) > max_calls:
            raise ValueError("graph tool call budget exhausted")
        if name == "graph_search":
            return self._graph_search(parameters, info, state)
        if name == "inspect_entity":
            snapshot = str(info.get("graph_snapshot", "kqapro-v1"))
            entity_id = str(parameters["entity_id"])
            return self._backend(snapshot).entity_info(entity_id).model_dump_json()
        if name == "text_search":
            if not bool(info.get("text_search_enabled", False)):
                raise ValueError("text search is not enabled for this graph snapshot")
            snapshot = str(info.get("graph_snapshot", "kqapro-v1"))
            passages = execute_text_search(
                self._backend(snapshot),
                str(parameters["query"]),
                limit=min(
                    max(1, _int_value(parameters.get("limit", 3))),
                    _int_value(info.get("max_text_search_results", 3)),
                ),
                max_chars=_int_value(info.get("max_passage_chars", 2_000)),
                trace_id=f"{info.get('task_id', 'solver')}:{state.get('calls', 0)}",
            )
            visible = set(_string_list(state.get("visible_entities", [])))
            visible.update(passage.page_id for passage in passages)
            state["visible_entities"] = sorted(visible)
            return json.dumps(
                [passage.model_dump(mode="json") for passage in passages],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        raise ValueError(f"unsupported solver tool: {name}")

    def step(self, infer_request: Any, response_choice: Any, current_turn: int) -> Any:
        del current_turn
        info = self._info(infer_request)
        if str(info.get("role", "solver")) != "solver":
            raise ValueError("ms-swift multi-turn scheduler is solver-only")
        state = self._state(infer_request, info)
        for call in _tool_calls(response_choice):
            function = getattr(call, "function", None)
            name = str(getattr(function, "name", ""))
            try:
                parameters = _parse_arguments(getattr(function, "arguments", "{}"))
                content = self._execute_tool(name, parameters, info, state)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state["invalid_calls"] = _int_value(state.get("invalid_calls", 0)) + 1
                content = json.dumps(
                    {
                        "error": {
                            "reason_code": "INVALID_TOOL_CALL",
                            "message": str(exc),
                        }
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            infer_request.messages.append({"role": "tool", "content": content})
        return infer_request


_register_data()
orms["graphtask_score"] = GraphTaskReward
if os.environ.get("INTERACTION_MODE", "graphscript") == "tool":
    try:
        from swift.plugin import multi_turns
    except (AssertionError, ImportError) as exc:  # pragma: no cover - training extra
        raise ImportError(
            "ms-swift tool mode requires its optional math_verify dependency"
        ) from exc
    multi_turns["graphtask_solver"] = GraphTaskSolverScheduler
