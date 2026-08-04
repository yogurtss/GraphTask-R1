from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.schema import TaskCertificate
from graphtask_r1.training.prompts import role_prompt


def export_role_dataset(
    tasks: list[TaskCertificate],
    output_path: Path,
    *,
    include_questioner: bool = True,
    include_solver: bool = True,
) -> int:
    """Export verl RLHFDataset rows with role-specific prompts and tool session kwargs."""
    rows: list[dict[str, object]] = []
    for index, task in enumerate(tasks):
        graph_snapshot = str(task.generation.get("graph_snapshot", "toy-v1"))
        topic_ids = [entity.entity_id for entity in task.topic_entities]
        common = {
            "graph_snapshot": graph_snapshot,
            "topic_entity_ids": topic_ids,
            "task_id": task.task_id,
            "index": index,
        }
        if include_questioner:
            payload = (
                "Start from these seed entities and construct a novel, executable challenge: "
                + ", ".join(topic_ids)
            )
            rows.append(
                {
                    "data_source": "graphtask/questioner",
                    "prompt": role_prompt("questioner", payload),
                    "ability": "graph_task_generation",
                    "reward_model": {"style": "rule", "ground_truth": "{}"},
                    "extra_info": {**common, "role": "questioner", "opponent_pass_rate": 0.5},
                    "agent_name": "tool_agent",
                    "tools_kwargs": {
                        name: {"create_kwargs": {**common, "role": "questioner"}}
                        for name in ("graph_search", "inspect_entity", "execute_program")
                    },
                }
            )
        if include_solver:
            rows.append(
                {
                    "data_source": "graphtask/solver",
                    "prompt": role_prompt(
                        "solver",
                        f"Question: {task.question}\nTopic entities: {', '.join(topic_ids)}",
                    ),
                    "ability": "graph_search_qa",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": task.gold_answers.model_dump_json(),
                    },
                    "extra_info": {**common, "role": "solver"},
                    "agent_name": "tool_agent",
                    "tools_kwargs": {
                        name: {"create_kwargs": {**common, "role": "solver"}}
                        for name in ("graph_search", "inspect_entity")
                    },
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    return len(rows)
