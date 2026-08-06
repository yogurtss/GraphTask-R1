from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.generation import compile_trace, validate_proposal
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import program_to_graphscript
from graphtask_r1.schema import RelationInfo, TaskCertificate, TaskProposal
from graphtask_r1.training.prompts import InteractionMode, role_prompt
from graphtask_r1.training.relations import require_catalog_covers_program
from graphtask_r1.utils import ProgressLogger


def _questioner_messages(
    task: TaskCertificate,
    *,
    interaction_mode: InteractionMode,
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
        )
    ]
    if interaction_mode == "graphscript":
        if len(topic_ids) != 1:
            raise ValueError("GraphScript v0.1 SFT requires exactly one topic entity")
        script = program_to_graphscript(task.program)
        messages.append(
            {"role": "assistant", "content": script.model_dump_json(by_alias=True)}
        )
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


def _solver_messages(
    task: TaskCertificate,
    backend: GraphBackend,
    *,
    seed: int,
    interaction_mode: InteractionMode,
    relation_catalog: tuple[RelationInfo, ...],
) -> list[dict[str, object]]:
    topic_ids = tuple(entity.entity_id for entity in task.topic_entities)
    messages: list[dict[str, object]] = [
        dict(value)
        for value in role_prompt(
            "solver",
            f"Question: {task.question}\nTopic entities: {', '.join(topic_ids)}",
            interaction_mode=interaction_mode,
            relation_catalog=relation_catalog,
        )
    ]
    if interaction_mode == "graphscript":
        if len(topic_ids) != 1:
            raise ValueError("GraphScript v0.1 SFT requires exactly one topic entity")
        script = program_to_graphscript(task.program)
        messages.append(
            {"role": "assistant", "content": script.model_dump_json(by_alias=True)}
        )
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
                        "function": {"name": tool_name, "arguments": call.arguments},
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
    tasks: list[TaskCertificate],
    output_path: Path,
    *,
    include_questioner: bool = True,
    include_solver: bool = True,
    seed: int = 42,
    interaction_mode: InteractionMode = "tool",
    relation_catalog: tuple[RelationInfo, ...] = (),
    relation_catalogs: Mapping[str, tuple[RelationInfo, ...]] | None = None,
) -> int:
    rows: list[dict[str, object]] = []
    backends: dict[str, GraphBackend] = {}
    progress = ProgressLogger("data.export_sft", total=len(tasks))
    progress.start(interaction_mode=interaction_mode)
    for index, task in enumerate(tasks):
        task_catalog = (relation_catalogs or {}).get(task.graph_snapshot, relation_catalog)
        if interaction_mode == "graphscript" and not task_catalog:
            raise ValueError(
                f"graphscript SFT export requires a relation catalog for {task.graph_snapshot}"
            )
        if interaction_mode == "graphscript":
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
            rows.append(
                {
                    "messages": _questioner_messages(
                        task,
                        interaction_mode=interaction_mode,
                        relation_catalog=task_catalog,
                    ),
                    "role": "questioner",
                    "task_id": task.task_id,
                    "interaction_mode": interaction_mode,
                }
            )
        if include_solver:
            if task.graph_snapshot not in backends:
                backends[task.graph_snapshot] = backend_from_snapshot(task.graph_snapshot)
            backend = backends[task.graph_snapshot]
            rows.append(
                {
                    "messages": _solver_messages(
                        task,
                        backend,
                        seed=seed + index,
                        interaction_mode=interaction_mode,
                        relation_catalog=task_catalog,
                    ),
                    "role": "solver",
                    "task_id": task.task_id,
                    "interaction_mode": interaction_mode,
                }
            )
        progress.update(index + 1, rows=len(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    progress.finish(len(tasks), rows=len(rows), output=str(output_path))
    return len(rows)
