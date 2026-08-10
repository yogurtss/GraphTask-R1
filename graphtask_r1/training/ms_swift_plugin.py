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

from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.training.json_compat import to_json_compatible
from graphtask_r1.training.ms_swift_data import convert_rl_row, convert_sft_row
from graphtask_r1.training.verl_reward import compute_score

try:
    from swift.llm.dataset import DatasetMeta, RowPreprocessor, register_dataset
    from swift.plugin import ORM, multi_turns, orms
    from swift.plugin.multi_turn import MultiTurnScheduler
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

        async def score_one(index: int) -> dict[str, float]:
            info = to_json_compatible(infos[index])
            if not isinstance(info, dict):
                raise ValueError("extra_info reward column must contain objects")
            return await compute_score(
                str(sources[index]),
                completions[index],
                str(truths[index]),
                info,
            )

        async def score_all() -> list[dict[str, float]]:
            return list(await asyncio.gather(*(score_one(index) for index in range(size))))

        results = asyncio.run(score_all())
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for result in results:
            for name, value in result.items():
                sums[name] += float(value)
                counts[name] += 1
        components = {
            name: sums[name] / counts[name]
            for name in sorted(sums)
            if counts[name]
        }
        logger.info(
            json.dumps(
                {
                    "event": "graphtask_reward_components",
                    "rank": os.environ.get("RANK", "0"),
                    "samples": size,
                    "means": components,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
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


class GraphTaskSolverScheduler(MultiTurnScheduler):  # type: ignore[misc]
    """Instance-scoped Hermes tool scheduler for Solver-only KQA Pro rollout."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
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

    def check_finished(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> bool:
        if super().check_finished(infer_request, response_choice, current_turn):
            return True
        return not _tool_calls(response_choice)

    def _graph_search(
        self,
        parameters: Mapping[str, object],
        info: Mapping[str, object],
        state: dict[str, object],
    ) -> str:
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
        snapshot = str(info.get("graph_snapshot", "kqapro-v1"))
        triples = self._backend(snapshot).neighbors(
            entity_ids,
            direction=str(parameters.get("direction", "both")),
            relation_ids=relation_ids or None,
            limit=limit,
            trace_id=f"{info.get('task_id', 'solver')}:{state.get('calls', 0)}",
        )
        state["edge_visits"] = _int_value(state.get("edge_visits", 0)) + len(triples)
        visible = set(_string_list(state.get("visible_entities", [])))
        visible.update(
            value
            for triple in triples
            for value in (triple.subject, triple.object)
        )
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
        raise ValueError(f"unsupported solver tool: {name}")

    def step(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> Any:
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
multi_turns["graphtask_solver"] = GraphTaskSolverScheduler
