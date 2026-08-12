from __future__ import annotations

import json
from collections.abc import Iterable, Sized
from pathlib import Path

from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    Count,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    QueryAttribute,
    QueryRelation,
    RelationInfo,
    SelectAmong,
    SelectBetween,
    TaskCertificate,
    TaskTrainingRecord,
    Union,
)
from graphtask_r1.utils import ProgressLogger, write_json


def program_relations(program: Program) -> frozenset[str]:
    if isinstance(program, Hop):
        return program_relations(program.input) | {program.relation}
    if isinstance(program, FilterLiteral):
        return program_relations(program.input) | {program.relation}
    if isinstance(program, QueryAttribute | SelectAmong):
        return program_relations(program.input) | {program.attribute}
    if isinstance(program, QueryRelation):
        return program_relations(program.subject) | program_relations(program.object)
    if isinstance(program, SelectBetween):
        return (
            program_relations(program.left) | program_relations(program.right) | {program.attribute}
        )
    if isinstance(program, Intersect | Union):
        return frozenset().union(*(program_relations(branch) for branch in program.inputs))
    if isinstance(program, FilterType | Count):
        return program_relations(program.input)
    return frozenset()


def require_catalog_covers_program(
    program: Program,
    relations: tuple[RelationInfo, ...],
    *,
    context: str,
) -> None:
    catalog_ids = {relation.relation_id for relation in relations}
    missing = sorted(program_relations(program) - catalog_ids)
    if missing:
        raise ValueError(
            f"{context} relation catalog is missing program relations: {', '.join(missing)}"
        )


def build_relation_catalog(
    tasks: Iterable[TaskCertificate | TaskTrainingRecord],
    backend: GraphBackend,
    output_path: Path,
    *,
    total: int | None = None,
    include_graph_schema: bool = False,
) -> tuple[RelationInfo, ...]:
    if total is None and isinstance(tasks, Sized):
        total = len(tasks)
    relation_ids_set: set[str] = set()
    scan = ProgressLogger("data.build_relation_catalog.scan", total=total)
    scan.start()
    task_count = 0
    for task_count, task in enumerate(tasks, start=1):
        relation_ids_set.update(program_relations(task.program))
        scan.update(task_count, relations=len(relation_ids_set))
    scan.finish(task_count, relations=len(relation_ids_set))
    if include_graph_schema:
        all_relation_infos = getattr(backend, "all_relation_infos", None)
        if not callable(all_relation_infos):
            raise ValueError(
                "graph-schema relation catalog is unavailable for this graph backend; "
                "use --scope tasks"
            )
        relations = tuple(all_relation_infos())
        schema_ids = {relation.relation_id for relation in relations}
        missing_from_graph = sorted(relation_ids_set - schema_ids)
        if missing_from_graph:
            raise ValueError(
                "task programs reference relations absent from the graph schema: "
                + ", ".join(missing_from_graph)
            )
    else:
        relations = tuple(
            backend.relation_info(relation_id) for relation_id in sorted(relation_ids_set)
        )
    progress = ProgressLogger("data.build_relation_catalog", total=len(relations))
    progress.start(tasks=task_count, scope="graph" if include_graph_schema else "tasks")
    for index in range(len(relations)):
        progress.update(index + 1)
    write_json(output_path, [relation.model_dump(mode="json") for relation in relations])
    progress.finish(
        len(relations),
        output=str(output_path),
        scope="graph" if include_graph_schema else "tasks",
        task_relations=len(relation_ids_set),
    )
    return relations


def load_relation_catalog(path: Path | None) -> tuple[RelationInfo, ...]:
    if path is None:
        return ()
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("relation catalog must be a JSON list")
    relations = tuple(RelationInfo.model_validate(value) for value in raw)
    if len({relation.relation_id for relation in relations}) != len(relations):
        raise ValueError("relation catalog contains duplicate relation IDs")
    return tuple(sorted(relations, key=lambda value: value.relation_id))
