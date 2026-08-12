from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path
from typing import Literal

import pyarrow as pa

from graphtask_r1.generation import compile_trace, validate_proposal
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import graphscript_operators, program_to_graphscript
from graphtask_r1.schema import RelationInfo, TaskCertificate, TaskProposal, TaskTrainingRecord
from graphtask_r1.training.prompts import GraphScriptVersion, InteractionMode, role_prompt
from graphtask_r1.training.relations import require_catalog_covers_program
from graphtask_r1.utils import ParquetRowWriter, ProgressLogger

TrainingTask = TaskCertificate | TaskTrainingRecord

SFT_SCHEMA = pa.schema(
    [
        (
            "messages",
            pa.list_(
                pa.struct(
                    [
                        ("role", pa.string()),
                        ("content", pa.string()),
                        ("name", pa.string()),
                        ("tool_call_id", pa.string()),
                        (
                            "tool_calls",
                            pa.list_(
                                pa.struct(
                                    [
                                        ("id", pa.string()),
                                        ("type", pa.string()),
                                        (
                                            "function",
                                            pa.struct(
                                                [
                                                    ("name", pa.string()),
                                                    ("arguments", pa.string()),
                                                ]
                                            ),
                                        ),
                                    ]
                                )
                            ),
                        ),
                    ]
                )
            ),
        ),
        ("role", pa.string()),
        ("task_id", pa.string()),
        ("interaction_mode", pa.string()),
        ("graphscript_version", pa.string()),
        ("operator_set", pa.list_(pa.string())),
    ]
)


def _questioner_messages(
    task: TrainingTask,
    *,
    interaction_mode: InteractionMode,
    graphscript_version: GraphScriptVersion,
    relation_catalog: tuple[RelationInfo, ...],
) -> list[dict[str, object]]:
    topic_ids = tuple(entity.entity_id for entity in task.topic_entities)
    proposal = TaskProposal(topic_entities=topic_ids, program=task.program)
    messages: list[dict[str, object]] = [
        dict(value)
        for value in role_prompt(
            "questioner",
            "Explore from these seed entities and construct one certified task: "
            + ", ".join(topic_ids),
            interaction_mode=interaction_mode,
            relation_catalog=relation_catalog,
            graphscript_version=graphscript_version,
        )
    ]
    if interaction_mode == "graphscript":
        if graphscript_version == "0.1" and len(topic_ids) != 1:
            raise ValueError("GraphScript v0.1 SFT requires exactly one topic entity")
        script = program_to_graphscript(task.program, version=graphscript_version)
        messages.append({"role": "assistant", "content": script.model_dump_json(by_alias=True)})
        return messages
    messages.append(
        {
            "role": "assistant",
            "content": "<task>" + proposal.model_dump_json(exclude_none=False) + "</task>",
        }
    )
    return messages


def _observation_content(observation: object) -> str:
    from graphtask_r1.schema import Observation

    value = Observation.model_validate(observation)
    return value.model_dump_json(exclude={"step"})


def _entity_reference(
    backend: GraphBackend, entity_id: str
) -> tuple[str, Literal["id", "exact", "search"]]:
    """Prefer a unique human-readable label; retain the ID when labels are ambiguous."""

    label = backend.entity_info(entity_id).label
    resolve = getattr(backend, "resolve_entities", None)
    if callable(resolve):
        try:
            if tuple(resolve(label, match="exact", limit=2)) == (entity_id,):
                return label, "exact"
        except (TypeError, ValueError):
            pass
    return entity_id, "id"


def _solver_messages(
    task: TrainingTask,
    backend: GraphBackend,
    *,
    seed: int,
    interaction_mode: InteractionMode,
    graphscript_version: GraphScriptVersion,
    relation_catalog: tuple[RelationInfo, ...],
    entity_reference_cache: dict[str, tuple[str, Literal["id", "exact", "search"]]],
) -> list[dict[str, object]]:
    topic_ids = tuple(entity.entity_id for entity in task.topic_entities)
    messages: list[dict[str, object]] = [
        dict(value)
        for value in role_prompt(
            "solver",
            (
                f"Question: {task.question}\nTopic entities: {', '.join(topic_ids)}"
                if graphscript_version == "0.1"
                else f"Question: {task.question}"
            ),
            interaction_mode=interaction_mode,
            relation_catalog=relation_catalog,
            graphscript_version=graphscript_version,
        )
    ]
    if interaction_mode == "graphscript":
        if graphscript_version == "0.1" and len(topic_ids) != 1:
            raise ValueError("GraphScript v0.1 SFT requires exactly one topic entity")
        def cached_entity_reference(
            entity_id: str,
        ) -> tuple[str, Literal["id", "exact", "search"]]:
            if entity_id not in entity_reference_cache:
                entity_reference_cache[entity_id] = _entity_reference(backend, entity_id)
            return entity_reference_cache[entity_id]

        script = program_to_graphscript(
            task.program,
            version=graphscript_version,
            entity_reference=cached_entity_reference,
        )
        messages.append({"role": "assistant", "content": script.model_dump_json(by_alias=True)})
        return messages
    trace = compile_trace(task.task_id, task.question, task.program, backend, seed=seed)
    for index, call in enumerate(trace.calls):
        if call.name == "final_answer":
            messages.append(
                {
                    "role": "assistant",
                    "content": "<answer>"
                    + json.dumps(call.arguments["answers"], ensure_ascii=False)
                    + "</answer>",
                }
            )
            continue
        tool_name = "graph_search" if call.name == "search" else call.name
        tool_call_id = call.trace_id
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "tool_call_id": tool_call_id,
                "content": _observation_content(trace.observations[index + 1]),
            }
        )
    return messages


def export_sft_dataset(
    tasks: Iterable[TrainingTask],
    output_path: Path,
    *,
    total: int | None = None,
    include_questioner: bool = True,
    include_solver: bool = True,
    seed: int = 42,
    interaction_mode: InteractionMode = "tool",
    graphscript_version: GraphScriptVersion = "0.1",
    relation_catalog: tuple[RelationInfo, ...] = (),
    relation_catalogs: Mapping[str, tuple[RelationInfo, ...]] | None = None,
) -> int:
    if total is None and isinstance(tasks, Sized):
        total = len(tasks)
    row_count = 0
    completed = 0
    backends: dict[str, GraphBackend] = {}
    entity_reference_caches: dict[
        str, dict[str, tuple[str, Literal["id", "exact", "search"]]]
    ] = {}
    progress = ProgressLogger("data.export_sft", total=total)
    progress.start(interaction_mode=interaction_mode)
    with ParquetRowWriter(output_path, schema=SFT_SCHEMA, batch_size=256) as writer:
        for index, task in enumerate(tasks):
            completed = index + 1
            task_catalog = (relation_catalogs or {}).get(task.graph_snapshot, relation_catalog)
            if (
                interaction_mode == "graphscript"
                and graphscript_version == "0.1"
                and not task_catalog
            ):
                raise ValueError(
                    f"graphscript SFT export requires a relation catalog for {task.graph_snapshot}"
                )
            if interaction_mode == "graphscript":
                if (include_questioner and task.topic_entities) or graphscript_version == "0.1":
                    validate_proposal(
                        TaskProposal(
                            topic_entities=tuple(
                                entity.entity_id for entity in task.topic_entities
                            ),
                            program=task.program,
                        )
                    )
                require_catalog_covers_program(
                    task.program,
                    task_catalog,
                    context=f"task {task.task_id}",
                )
            if include_questioner and task.topic_entities:
                writer.write(
                    {
                        "messages": _questioner_messages(
                            task,
                            interaction_mode=interaction_mode,
                            graphscript_version=graphscript_version,
                            relation_catalog=task_catalog,
                        ),
                        "role": "questioner",
                        "task_id": task.task_id,
                        "interaction_mode": interaction_mode,
                        "graphscript_version": graphscript_version,
                        "operator_set": list(graphscript_operators(graphscript_version)),
                    }
                )
                row_count += 1
            if include_solver:
                if task.graph_snapshot not in backends:
                    backends[task.graph_snapshot] = backend_from_snapshot(task.graph_snapshot)
                    entity_reference_caches[task.graph_snapshot] = {}
                backend = backends[task.graph_snapshot]
                writer.write(
                    {
                        "messages": _solver_messages(
                            task,
                            backend,
                            seed=seed + index,
                            interaction_mode=interaction_mode,
                            graphscript_version=graphscript_version,
                            relation_catalog=task_catalog,
                            entity_reference_cache=entity_reference_caches[
                                task.graph_snapshot
                            ],
                        ),
                        "role": "solver",
                        "task_id": task.task_id,
                        "interaction_mode": interaction_mode,
                        "graphscript_version": graphscript_version,
                        "operator_set": list(graphscript_operators(graphscript_version)),
                    }
                )
                row_count += 1
            progress.update(completed, rows=row_count)
    progress.finish(completed, rows=row_count, output=str(output_path))
    return row_count
