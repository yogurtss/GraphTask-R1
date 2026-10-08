"""ms-swift v3.10.3 runtime plugin for existing GraphTask Parquet files.

This module is imported by ``swift`` through ``--external_plugins``. Importing the rest of
GraphTask does not require ms-swift to be installed.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from graphtask_r1.envs.graph_query import execute_compact_query
from graphtask_r1.envs.text_search import execute_text_search
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import (
    BackendEvidenceRetriever,
    CounterfactualEvidenceRetriever,
    EvidenceRetriever,
)
from graphtask_r1.schema import PassageHit, parse_program
from graphtask_r1.training.json_compat import to_json_compatible
from graphtask_r1.training.ms_swift_data import convert_rl_row, convert_sft_row
from graphtask_r1.training.ms_swift_reward import compute_score
from graphtask_r1.training.opponent import OpponentTimeout, OpponentUnavailable
from graphtask_r1.training.response_normalization import normalize_reward_response

try:
    from swift.llm.dataset import DatasetMeta, RowPreprocessor, register_dataset
    from swift.plugin import ORM, orms
    from swift.plugin.multi_turn import MultiTurnScheduler
except ImportError as exc:  # pragma: no cover - exercised on the training server
    raise ImportError(
        "Install the pinned ms-swift environment before loading the GraphTask plugin"
    ) from exc


logger = logging.getLogger(__name__)


def _destroy_distributed_process_group() -> None:
    """Release NCCL resources before Python tears down CUDA objects."""

    try:
        import torch.distributed as distributed
    except ImportError:  # pragma: no cover - torch is required by ms-swift
        return
    if distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()


atexit.register(_destroy_distributed_process_group)


def _reward_completion(text: str) -> str:
    """Remove framework thinking text without repairing actor output."""

    return normalize_reward_response(text)


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
    val_path = os.environ.get("GRAPHTASK_MS_SWIFT_VAL_DATA", "")
    if not train_path:
        raise ValueError("GRAPHTASK_MS_SWIFT_TRAIN_DATA is required")
    for path in {train_path, *([val_path] if val_path else [])}:
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
    if val_path:
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


def _reward_group_key(prompt_id: object, info: Mapping[str, object], index: int) -> str:
    """Return ms-swift's GRPO group id, with stable fallbacks for older rows."""

    if prompt_id is not None and str(prompt_id):
        return str(prompt_id)
    task_id = info.get("task_id")
    if task_id is not None and str(task_id):
        return str(task_id)
    # Old/custom datasets may provide neither identifier. Keep those samples
    # independent instead of accidentally neutralizing every anonymous row.
    return f"__graphtask_ungrouped__:{os.environ.get('RANK', '0')}:{index}"


def _score_parallelism(infos: Sequence[Mapping[str, object]], size: int) -> int:
    """Bound in-flight reward calls by the frozen-opponent completion budget."""

    limits: list[int] = []
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    for info in infos:
        raw_max_concurrency = info.get("opponent_max_concurrency")
        if raw_max_concurrency is None:
            continue
        max_concurrency = int(cast(Any, raw_max_concurrency))
        samples = int(cast(Any, info.get("opponent_samples", 4)))
        if max_concurrency <= 0:
            raise ValueError("opponent_max_concurrency must be positive")
        if samples <= 0:
            raise ValueError("opponent_samples must be positive")
        limits.append(max(1, max_concurrency // (world_size * samples)))
    # Rows produced before the concurrency field was introduced retain the
    # previous batch-wide behavior.
    return min(limits) if limits else max(1, size)


def _distributed_reward_state(
    local_groups: set[str],
    local_error: Exception | None,
) -> tuple[set[str], str | None]:
    """Synchronize failed groups and scorer errors without stranding another rank."""

    try:
        import torch.distributed as distributed
    except ImportError:  # pragma: no cover - torch is required by ms-swift
        return set(local_groups), None
    if not distributed.is_available() or not distributed.is_initialized():
        return set(local_groups), None
    payload: dict[str, object] = {
        "error": (
            {
                "rank": os.environ.get("RANK", "unknown"),
                "type": type(local_error).__name__,
                "message": str(local_error),
            }
            if local_error is not None
            else None
        ),
        "unavailable_groups": sorted(local_groups),
    }
    gathered: list[object] = [None] * distributed.get_world_size()
    distributed.all_gather_object(gathered, payload)
    merged = set(local_groups)
    remote_error: str | None = None
    for rank_index, rank_state in enumerate(gathered):
        if not isinstance(rank_state, dict):
            raise TypeError("distributed reward state must be an object")
        rank_groups = rank_state.get("unavailable_groups")
        if not isinstance(rank_groups, list | tuple | set | frozenset):
            raise TypeError("distributed reward timeout groups must be a sequence")
        merged.update(str(group) for group in rank_groups)
        error = rank_state.get("error")
        if error is not None and not isinstance(error, dict):
            raise TypeError("distributed reward error state must be an object")
        if error is not None and remote_error is None:
            remote_error = (
                "reward scoring failed on rank "
                f"{error.get('rank', rank_index)}: "
                f"{error.get('type', 'Exception')}: {error.get('message', '')}"
            )
    return merged, remote_error


def _distributed_is_active() -> bool:
    try:
        import torch.distributed as distributed
    except ImportError:  # pragma: no cover - torch is required by ms-swift
        return False
    return bool(distributed.is_available() and distributed.is_initialized())


def _float_components(result: Mapping[str, object]) -> dict[str, float]:
    """Normalize component types while surfacing invalid scorer output."""

    components = {str(name): float(cast(Any, value)) for name, value in result.items()}
    non_finite = [name for name, value in components.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"reward components must be finite: {sorted(non_finite)}")
    if "score" not in components:
        raise ValueError("reward result must contain score")
    return components


class GraphTaskReward(ORM):  # type: ignore[misc]
    """Return the total reward and emit all auditable components as structured logs."""

    def __init__(self) -> None:
        super().__init__()
        self._metrics_sequence = 0
        self._backends: dict[str, GraphBackend] = {}
        metrics_dir = os.environ.get("GRAPHTASK_REWARD_METRICS_DIR")
        rank = os.environ.get("RANK", "0")
        safe_rank = rank if rank.isdigit() else "unknown"
        self._metrics_path = (
            Path(metrics_dir) / f"reward_components.rank-{safe_rank}.jsonl" if metrics_dir else None
        )

    def _backend(self, snapshot: str) -> GraphBackend:
        if snapshot not in self._backends:
            self._backends[snapshot] = backend_from_snapshot(snapshot)
        return self._backends[snapshot]

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
        raw_group_mode = os.environ.get("GRAPHTASK_RULE_GROUP_NEUTRALIZATION")
        if raw_group_mode not in {None, "true", "false"}:
            raise ValueError(
                "GRAPHTASK_RULE_GROUP_NEUTRALIZATION must be 'true' or 'false'"
            )
        configured_group_mode = (
            raw_group_mode == "true" if raw_group_mode is not None else None
        )
        try:
            size = len(completions)
            sources = _batch(data_source, size, default="graphtask/solver")
            truths = _batch(ground_truth, size, default="")
            infos = _batch(extra_info, size, default={})
            rollout_infos = _batch(kwargs.get("rollout_infos"), size, default={})
            prompt_ids = _batch(kwargs.get("prompt_id"), size, default=None)

            normalized_infos: list[dict[str, Any]] = []
            for index, raw_info in enumerate(infos):
                info = to_json_compatible(raw_info)
                if not isinstance(info, dict):
                    raise ValueError("extra_info reward column must contain objects")
                if (
                    str(sources[index]) == "graphtask/solver"
                    and info.get("solver_reward_variant") == "curriculum_v3"
                ) or str(sources[index]) == "graphtask/evidence_solver":
                    rollout = to_json_compatible(rollout_infos[index])
                    if not isinstance(rollout, dict):
                        raise ValueError(
                            "rollout_infos reward column must contain objects"
                        )
                    info["solver_rollout"] = {
                        "calls": 0,
                        "valid_calls": 0,
                        "invalid_calls": 0,
                        "edge_visits": 0,
                        "new_visible_entities": 0,
                        **rollout,
                    }
                normalized_infos.append(info)

            group_keys = [
                _reward_group_key(prompt_ids[index], normalized_infos[index], index)
                for index in range(size)
            ]
            rule_indices = {
                index
                for index, info in enumerate(normalized_infos)
                if str(sources[index]) == "graphtask/questioner"
                and info.get("questioner_reward_variant")
                == "rule_program_question_v1"
            }
            if (
                configured_group_mode is None
                and rule_indices
                and _distributed_is_active()
            ):
                raise ValueError(
                    "distributed Rule Questioner requires "
                    "GRAPHTASK_RULE_GROUP_NEUTRALIZATION=true"
                )
            group_mode = (
                configured_group_mode
                if configured_group_mode is not None
                else bool(rule_indices)
            )
            if configured_group_mode is False and rule_indices:
                raise ValueError(
                    "Rule Questioner rows require group neutralization to be enabled"
                )
        except Exception as exc:
            # Healthy ranks reach this same collective after local scoring. Reporting
            # preflight failures here prevents them from waiting forever at the group
            # neutralization collective.
            if configured_group_mode is True:
                _distributed_reward_state(set(), exc)
            raise

        async def score_one(index: int, semaphore: asyncio.Semaphore) -> dict[str, float]:
            info = normalized_infos[index]
            acquired = False
            bounded_opponent = (
                str(sources[index]) == "graphtask/questioner"
                and info.get("questioner_reward_variant") == "rule_program_question_v1"
            )
            timeout_s: float | None = None
            deadline: float | None = None
            if bounded_opponent:
                timeout_s = float(info.get("opponent_request_timeout_s", 180.0))
                if not math.isfinite(timeout_s) or timeout_s <= 0.0:
                    raise ValueError(
                        "opponent_request_timeout_s must be finite and positive"
                    )
                deadline = asyncio.get_running_loop().time() + timeout_s

            async def run_scoring() -> dict[str, float]:
                nonlocal acquired
                scoring_info = info
                if bounded_opponent:
                    async with semaphore:
                        acquired = True
                        assert deadline is not None
                        scoring_info = dict(info)
                        scoring_info["opponent_request_timeout_s"] = max(
                            1e-6,
                            deadline - asyncio.get_running_loop().time(),
                        )
                        scoring_info["_opponent_reward_deadline_monotonic_s"] = deadline
                        return await compute_score(
                            str(sources[index]),
                            _reward_completion(completions[index]),
                            str(truths[index]),
                            scoring_info,
                            backend=self._backend(
                                str(info.get("graph_snapshot", "toy-v1"))
                            ),
                        )
                return await compute_score(
                    str(sources[index]),
                    _reward_completion(completions[index]),
                    str(truths[index]),
                    scoring_info,
                    backend=self._backend(str(info.get("graph_snapshot", "toy-v1"))),
                )

            scoring_task = asyncio.create_task(run_scoring())
            try:
                if bounded_opponent:
                    assert timeout_s is not None
                    try:
                        done, _ = await asyncio.wait((scoring_task,), timeout=timeout_s)
                    except BaseException:
                        scoring_task.cancel()
                        await asyncio.gather(scoring_task, return_exceptions=True)
                        raise
                else:
                    done = {scoring_task}
                    await scoring_task
                if bounded_opponent and scoring_task not in done:
                    stage = "scoring" if acquired else "queue"
                    scoring_task.cancel()
                    await asyncio.gather(scoring_task, return_exceptions=True)
                    logger.warning(
                        "reward deadline exceeded; assigning neutral reward and continuing "
                        "task_id=%s stage=%s timeout_s=%s",
                        info.get("task_id", ""),
                        stage,
                        timeout_s,
                    )
                    result = {
                        "score": 0.0,
                        "raw_score": 0.0,
                        "opponent_required": 1.0,
                        "opponent_evaluated": 0.0,
                        "opponent_timeout": 1.0,
                        "opponent_unavailable": 1.0,
                        "reward_fallback": 1.0,
                        "reward_group_neutralized": 0.0,
                        "reject_opponent_timeout": 1.0,
                        f"opponent_timeout_stage_reward_{stage}": 1.0,
                    }
                else:
                    # Reading a completed task's result separately from the deadline
                    # lets an application-raised TimeoutError propagate as a code/runtime
                    # error instead of being mistaken for our scheduling deadline.
                    result = scoring_task.result()
            except OpponentUnavailable as exc:
                timed_out = isinstance(exc, OpponentTimeout)
                logger.warning(
                    "opponent unavailable; assigning neutral reward and continuing "
                    "task_id=%s timeout=%s error=%s",
                    info.get("task_id", ""),
                    timed_out,
                    exc,
                )
                result = {
                    "score": 0.0,
                    "raw_score": 0.0,
                    "opponent_required": 1.0,
                    "opponent_evaluated": 0.0,
                    "opponent_timeout": float(timed_out),
                    "opponent_unavailable": 1.0,
                    "reward_fallback": 1.0,
                    "reward_group_neutralized": 0.0,
                    (
                        "reject_opponent_timeout"
                        if timed_out
                        else "reject_opponent_unavailable"
                    ): 1.0,
                }
            return _float_components(result)

        async def score_all() -> list[dict[str, float]]:
            semaphore = asyncio.Semaphore(_score_parallelism(normalized_infos, size))
            return list(
                await asyncio.gather(*(score_one(index, semaphore) for index in range(size)))
            )

        results: list[dict[str, float]] = []
        local_error: Exception | None = None
        try:
            results = asyncio.run(score_all())
        except Exception as exc:
            local_error = exc
        local_unavailable_groups = {
            group_keys[index]
            for index, result in enumerate(results)
            if index in rule_indices
            and (
                result.get("opponent_timeout", 0.0) > 0.0
                or result.get("opponent_unavailable", 0.0) > 0.0
            )
        }
        unavailable_groups: set[str] = set()
        distributed_error: str | None = None
        if group_mode:
            unavailable_groups, distributed_error = _distributed_reward_state(
                local_unavailable_groups,
                local_error,
            )
        if local_error is not None:
            raise local_error
        if distributed_error is not None:
            raise RuntimeError(distributed_error)
        for index, result in enumerate(results):
            if index not in rule_indices or group_keys[index] not in unavailable_groups:
                continue
            result["score_before_neutralization"] = float(result["score"])
            result["score"] = 0.0
            result["raw_score"] = 0.0
            result["reward_group_neutralized"] = 1.0
            result["reward_fallback"] = 1.0
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        role_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        role_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        role_samples: dict[str, int] = defaultdict(int)
        sample_components: list[dict[str, object]] = []
        for index, result in enumerate(results):
            metrics = dict(result)
            role_weight = float(normalized_infos[index].get("role_weight", 1.0))
            if "raw_score" in result:
                metrics["unweighted_score"] = float(result["raw_score"])
            elif role_weight:
                metrics["unweighted_score"] = float(result["score"]) / role_weight
            source = str(sources[index])
            role = source.rsplit("/", maxsplit=1)[-1]
            sample_components.append(
                {
                    "batch_index": index,
                    "role": role,
                    "task_id": str(normalized_infos[index].get("task_id", "")),
                    "reason_codes": sorted(
                        name.removeprefix("reject_").upper()
                        for name, value in metrics.items()
                        if name.startswith("reject_") and float(value) > 0.0
                    ),
                    "components": {name: float(value) for name, value in sorted(metrics.items())},
                }
            )
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
            "rl_algorithm": os.environ.get("RL_ALGORITHM", "grpo"),
            "rank": os.environ.get("RANK", "0"),
            "world_size": os.environ.get("WORLD_SIZE", "1"),
            "samples": size,
            "means": components,
            "roles": roles,
            "sample_components": sample_components,
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


def _record_evidence_passages(
    state: dict[str, object], op: str, passages: list[dict[str, object]]
) -> None:
    existing = state.get("observed_passages", [])
    if not isinstance(existing, list):
        raise ValueError("invalid observed passage state")
    merged = {
        f"{value['page_id']}:{value['paragraph_id']}": value
        for value in existing
        if isinstance(value, dict)
    }
    merged.update({f"{value['page_id']}:{value['paragraph_id']}": value for value in passages})
    state["observed_passages"] = list(merged.values())
    actions = state.get("evidence_actions", [])
    if not isinstance(actions, list):
        raise ValueError("invalid evidence action state")
    actions.append({"op": op, "passages": passages})


def _passage_payload(passages: Sequence[PassageHit]) -> list[dict[str, object]]:
    """Expose the canonical follow-up action argument alongside every observation."""
    payload: list[dict[str, object]] = []
    for passage in passages:
        value = passage.model_dump(mode="json")
        value["passage_key"] = f"{passage.page_id}:{passage.paragraph_id}"
        payload.append(value)
    return payload


class GraphTaskSolverScheduler:
    """Instance-scoped Hermes tool scheduler for Solver-only graph and passage rollout."""

    def __init__(self, *args: object, max_turns: int | None = None, **kwargs: object) -> None:
        del args, kwargs
        self.max_turns = max_turns
        self._backends: dict[str, GraphBackend] = {}
        self._evidence_retrievers: dict[tuple[str, str], EvidenceRetriever] = {}

    def _backend(self, snapshot: str) -> GraphBackend:
        if snapshot not in self._backends:
            self._backends[snapshot] = backend_from_snapshot(snapshot)
        return self._backends[snapshot]

    def _evidence_retriever(
        self, snapshot: str, info: Mapping[str, object]
    ) -> EvidenceRetriever:
        path_value = str(
            info.get("evidence_reranker_path")
            or os.environ.get("EVIDENCE_RERANKER_PATH", "")
        )
        key = (snapshot, path_value)
        if key not in self._evidence_retrievers:
            backend = self._backend(snapshot)
            self._evidence_retrievers[key] = (
                CounterfactualEvidenceRetriever.from_path(backend, Path(path_value))
                if path_value
                else BackendEvidenceRetriever(backend)
            )
        return self._evidence_retrievers[key]

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
                "observed_passages": [],
                "selected_passage_keys": [],
                "evidence_actions": [],
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
            payload = _passage_payload(passages)
            _record_evidence_passages(state, "retrieve", payload)
            return json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if name == "expand_evidence":
            raw_observed = state.get("observed_passages", [])
            if not isinstance(raw_observed, list):
                raise ValueError("invalid observed passage state")
            observed = tuple(PassageHit.model_validate(value) for value in raw_observed)
            if not observed:
                raise ValueError("expand_evidence requires a prior text_search result")
            snapshot = str(info.get("graph_snapshot", "kilt-2019-08-01-v1"))
            expansion_query = str(parameters["query"])
            if str(info.get("evidence_reward_variant")) in {"ecp_v4", "ecp_v5"}:
                expansion_query = f"{info.get('question', '')} {expansion_query}".strip()
            passages = self._evidence_retriever(snapshot, info).expand(
                observed,
                expansion_query,
                limit=min(max(1, _int_value(parameters.get("limit", 3))), 10),
                trace_id=f"{info.get('task_id', 'solver')}:{state.get('calls', 0)}",
            )
            payload = _passage_payload(passages)
            _record_evidence_passages(state, "expand", payload)
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if name == "select_evidence":
            requested = _string_list(parameters.get("passage_keys", []))
            raw_observed = state.get("observed_passages", [])
            if not isinstance(raw_observed, list):
                raise ValueError("invalid observed passage state")
            observed_keys = {
                f"{value['page_id']}:{value['paragraph_id']}"
                for value in raw_observed
                if isinstance(value, dict)
            }
            selected = list(dict.fromkeys(key for key in requested if key in observed_keys))
            if not selected:
                raise ValueError("no selected passage key was observed")
            if str(info.get("evidence_reward_variant")) in {"ecp_v4", "ecp_v5"}:
                actions = state.get("evidence_actions", [])
                if not isinstance(actions, list):
                    raise ValueError("invalid evidence action state")
                retrieve_keys = _action_passage_keys(actions, "retrieve")
                expand_keys = _action_passage_keys(actions, "expand")
                selected_set = set(selected)
                if not expand_keys:
                    raise ValueError("select_evidence must causally follow expand_evidence")
                if not selected_set & retrieve_keys or not selected_set & expand_keys:
                    raise ValueError(
                        "selection must cover passages from both retrieve and expand"
                    )
            state["selected_passage_keys"] = selected
            actions = state.get("evidence_actions", [])
            if isinstance(actions, list):
                actions.append({"op": "select_evidence", "passage_keys": selected})
            return json.dumps(
                {"selected_passage_keys": selected},
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


class GraphTaskCurriculumSolverScheduler(GraphTaskSolverScheduler, MultiTurnScheduler):  # type: ignore[misc]
    """Tool scheduler that exposes cumulative process signals to curriculum rewards."""

    def __init__(self, *args: object, max_turns: int | None = None, **kwargs: object) -> None:
        MultiTurnScheduler.__init__(self, *args, max_turns=max_turns, **kwargs)
        self._backends: dict[str, GraphBackend] = {}
        self._evidence_retrievers: dict[tuple[str, str], EvidenceRetriever] = {}

    def _execute_tool(
        self,
        name: str,
        parameters: Mapping[str, object],
        info: Mapping[str, object],
        state: dict[str, object],
    ) -> str:
        if name == "execute_program":
            state["calls"] = _int_value(state.get("calls", 0)) + 1
            max_calls = _int_value(info.get("max_tool_calls", self.max_turns or 8))
            if _int_value(state["calls"]) > max_calls:
                raise ValueError("graph tool call budget exhausted")
            if str(info.get("role", "solver")) != "questioner":
                raise ValueError("execute_program is questioner-only")
            snapshot = str(info.get("graph_snapshot", "kqapro-v1"))
            program = parse_program(parameters["program"])
            return self._backend(snapshot).execute_program(program).model_dump_json()
        return super()._execute_tool(name, parameters, info, state)

    def check_finished(self, infer_request: Any, response_choice: Any, current_turn: int) -> bool:
        info = self._info(infer_request)
        if str(info.get("role")) != "evidence_solver":
            return super().check_finished(infer_request, response_choice, current_turn)
        if getattr(response_choice, "finish_reason", None) == "length":
            return True
        if self.max_turns is not None and current_turn >= self.max_turns:
            return True
        if _tool_calls(response_choice):
            return False
        state = self._state(infer_request, info)
        actions = state.get("evidence_actions", [])
        operation_types = (
            {
                str(value.get("op"))
                for value in actions
                if isinstance(value, Mapping)
            }
            if isinstance(actions, list)
            else set()
        )
        protocol_complete = bool(
            _string_list(state.get("selected_passage_keys", []))
            and {"retrieve", "expand", "select_evidence"} <= operation_types
        )
        if not protocol_complete:
            return False
        if str(info.get("evidence_reward_variant")) not in {"ecp_v4", "ecp_v5"}:
            return True
        return _causal_selection_complete(actions if isinstance(actions, list) else [])

    def step(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> dict[str, object]:
        del current_turn
        info = self._info(infer_request)
        if str(info.get("role", "solver")) not in {
            "questioner",
            "solver",
            "evidence_solver",
        }:
            raise ValueError("curriculum scheduler requires a Questioner or Solver role")
        state = self._state(infer_request, info)
        tool_calls = _tool_calls(response_choice)
        if str(info.get("role")) == "evidence_solver" and not tool_calls:
            state["early_answer_attempts"] = (
                _int_value(state.get("early_answer_attempts", 0)) + 1
            )
            actions = state.get("evidence_actions", [])
            operation_types = (
                {
                    str(value.get("op"))
                    for value in actions
                    if isinstance(value, Mapping)
                }
                if isinstance(actions, list)
                else set()
            )
            if "retrieve" not in operation_types:
                instruction = "Call text_search before answering."
            elif "expand" not in operation_types:
                instruction = (
                    "Call expand_evidence for the linked second hop before answering."
                )
            elif not _causal_selection_complete(
                actions if isinstance(actions, list) else []
            ):
                instruction = (
                    "After expand_evidence, call select_evidence with exact observed "
                    "passage_key values from both the retrieve and expand results."
                )
            else:
                instruction = "Complete the evidence protocol before answering."
            infer_request.messages.append(
                {
                    "role": "user",
                    "content": f"Evidence protocol incomplete. {instruction}",
                }
            )
        for call in tool_calls:
            function = getattr(call, "function", None)
            name = str(getattr(function, "name", ""))
            calls_before = _int_value(state.get("calls", 0))
            try:
                parameters = _parse_arguments(getattr(function, "arguments", "{}"))
                content = self._execute_tool(name, parameters, info, state)
                state["valid_calls"] = _int_value(state.get("valid_calls", 0)) + 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if _int_value(state.get("calls", 0)) == calls_before:
                    state["calls"] = calls_before + 1
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

        initial_entities = set(_string_list(info.get("topic_entity_ids", [])))
        visible_entities = set(_string_list(state.get("visible_entities", [])))
        rollout_infos: dict[str, object] = {
            "calls": _int_value(state.get("calls", 0)),
            "valid_calls": _int_value(state.get("valid_calls", 0)),
            "invalid_calls": _int_value(state.get("invalid_calls", 0)),
            "edge_visits": _int_value(state.get("edge_visits", 0)),
            "new_visible_entities": len(visible_entities - initial_entities),
        }
        if str(info.get("role")) == "evidence_solver":
            rollout_infos.update(
                {
                    "observed_passages": state.get("observed_passages", []),
                    "selected_passage_keys": state.get("selected_passage_keys", []),
                    "evidence_actions": state.get("evidence_actions", []),
                    "early_answer_attempts": _int_value(
                        state.get("early_answer_attempts", 0)
                    ),
                }
            )
        return {"infer_request": infer_request, "rollout_infos": rollout_infos}


def _action_passage_keys(actions: Sequence[object], operation: str) -> set[str]:
    keys: set[str] = set()
    for action in actions:
        if not isinstance(action, Mapping) or str(action.get("op")) != operation:
            continue
        passages = action.get("passages", [])
        if not isinstance(passages, Sequence) or isinstance(passages, str | bytes):
            continue
        for passage in passages:
            if not isinstance(passage, Mapping):
                continue
            key = passage.get("passage_key")
            if isinstance(key, str):
                keys.add(key)
    return keys


def _causal_selection_complete(actions: Sequence[object]) -> bool:
    expand_indices = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, Mapping) and str(action.get("op")) == "expand"
    ]
    select_indices = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, Mapping) and str(action.get("op")) == "select_evidence"
    ]
    if not expand_indices or not select_indices or select_indices[-1] < expand_indices[-1]:
        return False
    selected = next(
        (
            set(_string_list(action.get("passage_keys", [])))
            for action in reversed(actions)
            if isinstance(action, Mapping) and str(action.get("op")) == "select_evidence"
        ),
        set(),
    )
    return bool(
        selected & _action_passage_keys(actions, "retrieve")
        and selected & _action_passage_keys(actions, "expand")
    )


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
    multi_turns["graphtask_curriculum_solver"] = GraphTaskCurriculumSolverScheduler
