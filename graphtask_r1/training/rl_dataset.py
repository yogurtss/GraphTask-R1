from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.generation import validate_proposal
from graphtask_r1.graphscript import graphscript_operators
from graphtask_r1.schema import RelationInfo, TaskCertificate, TaskProposal
from graphtask_r1.training.prompts import GraphScriptVersion, InteractionMode, role_prompt
from graphtask_r1.training.relations import require_catalog_covers_program
from graphtask_r1.utils import ProgressLogger


def export_role_dataset(
    tasks: list[TaskCertificate],
    output_path: Path,
    *,
    include_questioner: bool = True,
    include_solver: bool = True,
    opponent_url: str | None = None,
    opponent_samples: int = 8,
    questioner_weight: float = 0.35,
    solver_weight: float = 0.65,
    interaction_mode: InteractionMode = "tool",
    graphscript_version: GraphScriptVersion = "0.1",
    relation_catalog: tuple[RelationInfo, ...] = (),
    relation_catalogs: Mapping[str, tuple[RelationInfo, ...]] | None = None,
    max_follow_limit: int = 100,
    max_edge_visits: int = 200,
    max_returned_entities: int = 1_000,
    program_profile: str = "full",
) -> int:
    """Export role-specific RL rows consumed directly by the ms-swift preprocessor."""
    rows: list[dict[str, object]] = []
    progress = ProgressLogger("data.export_rl", total=len(tasks))
    progress.start(
        interaction_mode=interaction_mode,
        graphscript_version=graphscript_version,
        program_profile=program_profile,
    )
    for index, task in enumerate(tasks):
        graph_snapshot = task.graph_snapshot
        task_catalog = (relation_catalogs or {}).get(graph_snapshot, relation_catalog)
        if interaction_mode == "graphscript" and graphscript_version == "0.1" and not task_catalog:
            raise ValueError(
                f"GraphScript RL export requires a relation catalog for {graph_snapshot}"
            )
        if program_profile in {"graphscript_v0_1", "graphscript_v0_2"}:
            if program_profile == "graphscript_v0_1" and not task_catalog:
                raise ValueError(
                    f"comparison RL export requires a relation catalog for {graph_snapshot}"
                )
            if program_profile == "graphscript_v0_1" or (
                include_questioner and task.topic_entities
            ):
                validate_proposal(
                    TaskProposal(
                        topic_entities=tuple(entity.entity_id for entity in task.topic_entities),
                        program=task.program,
                    )
                )
            require_catalog_covers_program(
                task.program,
                task_catalog,
                context=f"task {task.task_id}",
            )
        topic_ids = [entity.entity_id for entity in task.topic_entities]
        common = {
            "graph_snapshot": graph_snapshot,
            "topic_entity_ids": topic_ids,
            "task_id": task.task_id,
            "index": index,
            "interaction_mode": interaction_mode,
            "graphscript_version": graphscript_version,
            "operator_set": list(graphscript_operators(graphscript_version)),
            "allowed_relations": [value.relation_id for value in task_catalog],
            "max_follow_limit": max_follow_limit,
            "max_edge_visits": max_edge_visits,
            "max_returned_entities": max_returned_entities,
            "program_profile": program_profile,
            "text_search_enabled": graph_snapshot.startswith("kilt-"),
            "max_text_search_results": 3,
            "max_passage_chars": 2_000,
        }
        if include_questioner:
            questioner_extra = {
                **common,
                "role": "questioner",
                "role_weight": questioner_weight,
                "opponent_url": opponent_url,
                "opponent_samples": opponent_samples,
            }
            payload = (
                "Start from these seed entities and construct a novel, executable challenge: "
                + ", ".join(topic_ids)
            )
            rows.append(
                {
                    "data_source": "graphtask/questioner",
                    "prompt": role_prompt(
                        "questioner",
                        payload,
                        interaction_mode=interaction_mode,
                        relation_catalog=task_catalog,
                        graphscript_version=graphscript_version,
                    ),
                    "ability": "graph_task_generation",
                    "reward_model": {"style": "rule", "ground_truth": "{}"},
                    "extra_info": questioner_extra,
                    "uid": f"questioner:{task.task_id}",
                }
            )
        if include_solver:
            solver_extra = {
                **common,
                "role": "solver",
                "role_weight": solver_weight,
            }
            rows.append(
                {
                    "data_source": "graphtask/solver",
                    "prompt": role_prompt(
                        "solver",
                        (
                            f"Question: {task.question}\nTopic entities: {', '.join(topic_ids)}"
                            if graphscript_version == "0.1"
                            else f"Question: {task.question}"
                        ),
                        interaction_mode=interaction_mode,
                        relation_catalog=task_catalog,
                        graphscript_version=graphscript_version,
                    ),
                    "ability": "graph_search_qa",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": task.gold_answers.model_dump_json(),
                    },
                    "extra_info": solver_extra,
                    "uid": f"solver:{task.task_id}",
                }
            )
        progress.update(index + 1, rows=len(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    progress.finish(len(tasks), rows=len(rows), output=str(output_path))
    return len(rows)
