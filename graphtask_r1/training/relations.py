from __future__ import annotations

import json
from pathlib import Path

from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    Count,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    RelationInfo,
    TaskCertificate,
    Union,
)
from graphtask_r1.utils import ProgressLogger, write_json


def program_relations(program: Program) -> frozenset[str]:
    if isinstance(program, Hop):
        return program_relations(program.input) | {program.relation}
    if isinstance(program, FilterLiteral):
        return program_relations(program.input) | {program.relation}
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
    tasks: list[TaskCertificate], backend: GraphBackend, output_path: Path
) -> tuple[RelationInfo, ...]:
    relation_ids = sorted({value for task in tasks for value in program_relations(task.program)})
    progress = ProgressLogger("data.build_relation_catalog", total=len(relation_ids))
    progress.start(tasks=len(tasks))
    relations_list: list[RelationInfo] = []
    for index, relation_id in enumerate(relation_ids):
        relations_list.append(backend.relation_info(relation_id))
        progress.update(index + 1)
    relations = tuple(relations_list)
    write_json(output_path, [relation.model_dump(mode="json") for relation in relations])
    progress.finish(len(relation_ids), output=str(output_path))
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
