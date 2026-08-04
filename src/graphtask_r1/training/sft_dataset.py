from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.generation import compile_trace
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.schema import TaskCertificate, TaskProposal
from graphtask_r1.training.prompts import role_prompt


def _questioner_messages(task: TaskCertificate) -> list[dict[str, object]]:
    topic_ids = tuple(entity.entity_id for entity in task.topic_entities)
    proposal = TaskProposal(topic_entities=topic_ids, program=task.program)
    messages: list[dict[str, object]] = [
        dict(value)
        for value in role_prompt(
            "questioner",
            "Explore from these seed entities and construct one certified task: "
            + ", ".join(topic_ids),
        )
    ]
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
    task: TaskCertificate, backend: GraphBackend, *, seed: int
) -> list[dict[str, object]]:
    topic_ids = tuple(entity.entity_id for entity in task.topic_entities)
    messages: list[dict[str, object]] = [
        dict(value)
        for value in role_prompt(
            "solver",
            f"Question: {task.question}\nTopic entities: {', '.join(topic_ids)}",
        )
    ]
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
) -> int:
    rows: list[dict[str, object]] = []
    backends: dict[str, GraphBackend] = {}
    for index, task in enumerate(tasks):
        if include_questioner and task.topic_entities:
            rows.append(
                {
                    "messages": _questioner_messages(task),
                    "role": "questioner",
                    "task_id": task.task_id,
                }
            )
        if include_solver:
            if task.graph_snapshot not in backends:
                backends[task.graph_snapshot] = backend_from_snapshot(task.graph_snapshot)
            backend = backends[task.graph_snapshot]
            rows.append(
                {
                    "messages": _solver_messages(task, backend, seed=seed + index),
                    "role": "solver",
                    "task_id": task.task_id,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    return len(rows)
