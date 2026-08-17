from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import graphscript_operators
from graphtask_r1.schema import RelationInfo, TaskCertificate, TaskTrainingRecord
from graphtask_r1.training.prompts import GraphScriptVersion, InteractionMode, role_prompt
from graphtask_r1.training.questioner_context import (
    build_questioner_seed_context,
    render_questioner_seed_payload,
)
from graphtask_r1.training.questioner_sampling import (
    questioner_task_stratum,
    select_questioner_tasks,
)
from graphtask_r1.utils import ProgressLogger, write_json


def merge_denylists(inputs: list[Path], output: Path) -> dict[str, int]:
    values = sorted({str(value) for path in inputs for value in json.loads(path.read_text())})
    write_json(output, values)
    return {"inputs": len(inputs), "entities": len(values)}


def sample_questioner_seeds(
    snapshot: str,
    output_path: Path,
    *,
    count: int,
    seed: int,
    exclude_path: Path | None = None,
    pool_limit: int = 100_000,
    min_degree: int = 2,
    max_degree: int = 100,
    opponent_url: str | None = None,
    opponent_samples: int = 8,
    interaction_mode: InteractionMode = "tool",
    graphscript_version: GraphScriptVersion = "0.3",
    relation_catalog: tuple[RelationInfo, ...] = (),
    max_follow_limit: int = 100,
    max_edge_visits: int = 200,
    max_returned_entities: int = 1_000,
    max_seed_neighbor_facts: int = 200,
    max_seed_relations: int = 64,
) -> dict[str, int | str]:
    if interaction_mode == "graphscript" and not relation_catalog:
        raise ValueError("graphscript seed export requires a non-empty relation catalog")
    backend = backend_from_snapshot(snapshot)
    excluded = set(json.loads(exclude_path.read_text())) if exclude_path else set()
    candidates = [
        value
        for value in backend.all_entities(limit=pool_limit)
        if value not in excluded and not value.startswith(("common.", "type.", "base.", "user."))
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: list[str] = []
    progress = ProgressLogger("data.sample_seeds.filter", total=len(candidates))
    progress.start(requested=count, min_degree=min_degree, max_degree=max_degree)
    scanned = 0
    for scanned, entity_id in enumerate(candidates, start=1):
        degree = len(backend.neighbors([entity_id], direction="both", limit=max_degree + 1))
        if min_degree <= degree <= max_degree:
            selected.append(entity_id)
        progress.update(scanned, selected=len(selected))
        if len(selected) >= count:
            break
    progress.finish(scanned, selected=len(selected), requested=count)
    rows = []
    allowed_relations = frozenset(value.relation_id for value in relation_catalog)
    for index, entity_id in enumerate(selected):
        seed_context = build_questioner_seed_context(
            backend,
            [entity_id],
            allowed_relations=allowed_relations,
            max_neighbor_facts=max_seed_neighbor_facts,
            max_relation_ids=max_seed_relations,
        )
        common = {
            "graph_snapshot": snapshot,
            "topic_entity_ids": [entity_id],
            "task_id": f"seed-{seed}-{index:08d}",
            "index": index,
            "role": "questioner",
            "role_weight": 0.35,
            "opponent_url": opponent_url,
            "opponent_samples": opponent_samples,
            "interaction_mode": interaction_mode,
            "graphscript_version": graphscript_version,
            "operator_set": list(graphscript_operators(graphscript_version)),
            "allowed_relations": [value.relation_id for value in relation_catalog],
            "max_follow_limit": max_follow_limit,
            "max_edge_visits": max_edge_visits,
            "max_returned_entities": max_returned_entities,
            "program_profile": (
                f"graphscript_v{graphscript_version.replace('.', '_')}"
                if interaction_mode == "graphscript"
                else "full"
            ),
            "text_search_enabled": graphscript_version == "0.2",
            "seed_context": seed_context,
        }
        payload = (
            render_questioner_seed_payload(seed_context)
            if interaction_mode == "graphscript" and graphscript_version == "0.3"
            else f"Explore from this seed entity and construct one certified task: {entity_id}"
        )
        rows.append(
            {
                "data_source": "graphtask/questioner",
                "prompt": role_prompt(
                    "questioner",
                    payload,
                    interaction_mode=interaction_mode,
                    relation_catalog=relation_catalog,
                    graphscript_version=graphscript_version,
                ),
                "ability": "graph_task_generation",
                "reward_model": {"style": "rule", "ground_truth": "{}"},
                "extra_info": common,
                "uid": f"questioner:{common['task_id']}",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    metrics: dict[str, int | str] = {
        "snapshot": snapshot,
        "pool": len(candidates),
        "excluded": len(excluded),
        "requested": count,
        "selected": len(selected),
        "seed": seed,
        "interaction_mode": interaction_mode,
        "graphscript_version": graphscript_version,
        "max_seed_neighbor_facts": max_seed_neighbor_facts,
        "max_seed_relations": max_seed_relations,
    }
    write_json(output_path.with_suffix(".metrics.json"), metrics)
    return metrics


def export_questioner_task_seeds(
    tasks: Iterable[TaskCertificate | TaskTrainingRecord],
    output_path: Path,
    *,
    count: int,
    seed: int,
    relation_catalog: tuple[RelationInfo, ...],
    allow_oversample: bool = False,
    opponent_url: str | None = None,
    opponent_samples: int = 8,
    interaction_mode: InteractionMode = "graphscript",
    graphscript_version: GraphScriptVersion = "0.3",
    max_follow_limit: int = 100,
    max_edge_visits: int = 200,
    max_returned_entities: int = 1_000,
    max_seed_neighbor_facts: int = 200,
    max_seed_relations: int = 64,
    max_topic_entities: int | None = None,
) -> dict[str, object]:
    """Export non-leaking single/multi-root self-play seeds from certified tasks."""
    if interaction_mode == "graphscript" and not relation_catalog:
        raise ValueError("graphscript seed export requires a non-empty relation catalog")
    selected, sampling = select_questioner_tasks(
        tasks,
        count=count,
        seed=seed,
        allow_oversample=allow_oversample,
        max_topic_entities=max_topic_entities,
    )
    allowed_relations = frozenset(value.relation_id for value in relation_catalog)
    backends: dict[str, GraphBackend] = {}
    rows = []
    for index, task in enumerate(selected):
        if task.graph_snapshot not in backends:
            backends[task.graph_snapshot] = backend_from_snapshot(task.graph_snapshot)
        backend = backends[task.graph_snapshot]
        topic_ids = [entity.entity_id for entity in task.topic_entities]
        seed_context = build_questioner_seed_context(
            backend,
            topic_ids,
            allowed_relations=allowed_relations,
            max_neighbor_facts=max_seed_neighbor_facts,
            max_relation_ids=max_seed_relations,
        )
        task_id = f"task-seed-{seed}-{index:08d}"
        extra = {
            "graph_snapshot": task.graph_snapshot,
            "topic_entity_ids": topic_ids,
            "task_id": task_id,
            "source_task_id": task.task_id,
            "source_stratum": questioner_task_stratum(task),
            "index": index,
            "role": "questioner",
            "role_weight": 0.35,
            "opponent_url": opponent_url,
            "opponent_samples": opponent_samples,
            "interaction_mode": interaction_mode,
            "graphscript_version": graphscript_version,
            "operator_set": list(graphscript_operators(graphscript_version)),
            "allowed_relations": [value.relation_id for value in relation_catalog],
            "max_follow_limit": max_follow_limit,
            "max_edge_visits": max_edge_visits,
            "max_returned_entities": max_returned_entities,
            "program_profile": (
                f"graphscript_v{graphscript_version.replace('.', '_')}"
                if interaction_mode == "graphscript"
                else "full"
            ),
            "text_search_enabled": graphscript_version == "0.2",
            "seed_context": seed_context,
        }
        payload = (
            render_questioner_seed_payload(seed_context)
            if interaction_mode == "graphscript" and graphscript_version == "0.3"
            else "Explore from these seed entities and construct one certified task: "
            + ", ".join(topic_ids)
        )
        rows.append(
            {
                "data_source": "graphtask/questioner",
                "prompt": role_prompt(
                    "questioner",
                    payload,
                    interaction_mode=interaction_mode,
                    relation_catalog=relation_catalog,
                    graphscript_version=graphscript_version,
                ),
                "ability": "graph_task_generation",
                "reward_model": {"style": "rule", "ground_truth": "{}"},
                "extra_info": extra,
                "uid": f"questioner:{task_id}",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    metrics = {
        **sampling,
        "output": str(output_path),
        "selected_task_ids": [task.task_id for task in selected],
        "max_seed_neighbor_facts": max_seed_neighbor_facts,
        "max_seed_relations": max_seed_relations,
        "max_topic_entities": max_topic_entities,
    }
    write_json(output_path.with_suffix(".metrics.json"), metrics)
    summary_keys = (
        "scanned",
        "requested",
        "rootless_rejected",
        "too_many_roots_rejected",
        "eligible",
        "unique_selected",
        "repeated_rows",
        "shortfall",
        "selected",
        "source_strata",
        "selected_strata",
        "distribution_total_variation",
        "seed",
    )
    return {**{key: metrics[key] for key in summary_keys}, "output": str(output_path)}
