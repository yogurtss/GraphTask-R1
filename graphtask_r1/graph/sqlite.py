from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graphtask_r1.graph.materialize import materialize_program
from graphtask_r1.graph.overlay import GraphOverlay
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Count,
    Entity,
    EntityInfo,
    FilterLiteral,
    FilterType,
    GraphSlice,
    Hop,
    Intersect,
    Program,
    RelationInfo,
    Triple,
    Union,
    Witness,
    parse_program,
)

_PREFERRED_SQL_VARIABLE_LIMIT = 900


def _chunks(values: Sequence[str], size: int) -> list[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _compare(left: str, datatype: str, comparator: str, right: str | int | float) -> bool:
    if comparator == "contains":
        return str(right).casefold() in left.casefold()
    if datatype in {"quantity", "number", "year"}:
        try:
            lhs: Any = float(left)
            rhs: Any = float(right)
        except ValueError:
            lhs, rhs = left, str(right)
    else:
        lhs, rhs = left, str(right)
    if comparator == "eq":
        return bool(lhs == rhs)
    if comparator == "ne":
        return bool(lhs != rhs)
    if comparator == "lt":
        return bool(lhs < rhs)
    if comparator == "le":
        return bool(lhs <= rhs)
    if comparator == "gt":
        return bool(lhs > rhs)
    if comparator == "ge":
        return bool(lhs >= rhs)
    raise ValueError(f"invalid comparator: {comparator}")


class SQLiteGraphBackend:
    """Read-only indexed graph backend produced by the KQA Pro importer."""

    def __init__(
        self,
        path: Path,
        *,
        snapshot_id: str = "kqapro-v1",
        allow_cross_thread: bool = False,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path
        self.snapshot_id = snapshot_id
        self.connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            check_same_thread=not allow_cross_thread,
        )

    def _sql_variable_limit(self) -> int:
        runtime_limit = self.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        return max(2, min(_PREFERRED_SQL_VARIABLE_LIMIT, runtime_limit))

    def all_entities(self, *, limit: int) -> tuple[str, ...]:
        rows = self.connection.execute(
            "SELECT entity_id FROM entities ORDER BY entity_id LIMIT ?", (max(0, limit),)
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def neighbors(
        self,
        entity_ids: Sequence[str],
        *,
        direction: str,
        relation_ids: Sequence[str] | None = None,
        limit: int = 100,
        trace_id: str | None = None,
    ) -> list[Triple]:
        del trace_id
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"invalid direction: {direction}")
        if not entity_ids or limit <= 0:
            return []
        entities = sorted(set(entity_ids))
        relations = sorted(set(relation_ids or ()))
        bind_budget = self._sql_variable_limit() - 1  # reserve one variable for LIMIT
        if relations:
            entity_batch_size = max(1, bind_budget // 2)
            relation_batch_size = max(1, bind_budget - entity_batch_size)
            relation_batches: list[Sequence[str]] = _chunks(relations, relation_batch_size)
        else:
            entity_batch_size = bind_budget
            relation_batches = [()]
        columns = []
        if direction in {"out", "both"}:
            columns.append("subject")
        if direction in {"in", "both"}:
            columns.append("object")

        found: dict[tuple[str, str, str], Triple] = {}
        for column in columns:
            for entity_batch in _chunks(entities, entity_batch_size):
                entity_placeholders = ",".join("?" for _ in entity_batch)
                for relation_batch in relation_batches:
                    relation_clause = ""
                    if relation_batch:
                        relation_placeholders = ",".join("?" for _ in relation_batch)
                        relation_clause = f" AND relation IN ({relation_placeholders})"
                    rows = self.connection.execute(
                        "SELECT subject, relation, object FROM triples "
                        f"WHERE {column} IN ({entity_placeholders}){relation_clause} "
                        "ORDER BY subject, relation, object LIMIT ?",
                        (*entity_batch, *relation_batch, limit),
                    ).fetchall()
                    for row in rows:
                        triple = Triple(subject=row[0], relation=row[1], object=row[2])
                        found[triple.sort_key()] = triple
        return sorted(found.values(), key=Triple.sort_key)[:limit]

    def _execute_entities(self, program: Program) -> set[str]:
        if isinstance(program, AllEntities):
            return set(self.all_entities(limit=program.max_results))
        if isinstance(program, Entity):
            return {program.entity_id}
        if isinstance(program, Hop):
            inputs = self._execute_entities(program.input)
            edges = self.neighbors(
                sorted(inputs),
                direction=program.direction,
                relation_ids=[program.relation],
                limit=1_000_000,
            )
            return {edge.object if program.direction == "out" else edge.subject for edge in edges}
        if isinstance(program, Intersect):
            return set.intersection(*(self._execute_entities(branch) for branch in program.inputs))
        if isinstance(program, Union):
            return set().union(*(self._execute_entities(branch) for branch in program.inputs))
        if isinstance(program, FilterType):
            candidates = self._execute_entities(program.input)
            if not candidates:
                return set()
            found: set[str] = set()
            batch_size = self._sql_variable_limit() - 1
            for batch in _chunks(sorted(candidates), batch_size):
                placeholders = ",".join("?" for _ in batch)
                rows = self.connection.execute(
                    f"SELECT entity_id FROM entity_types WHERE entity_id IN ({placeholders}) "
                    "AND type_id = ?",
                    (*batch, program.type_id),
                ).fetchall()
                found.update(str(row[0]) for row in rows)
            return found
        if isinstance(program, FilterLiteral):
            candidates = self._execute_entities(program.input)
            if not candidates:
                return set()
            found = set()
            batch_size = self._sql_variable_limit() - 1
            for batch in _chunks(sorted(candidates), batch_size):
                placeholders = ",".join("?" for _ in batch)
                rows = self.connection.execute(
                    f"SELECT entity_id, value, datatype FROM attributes "
                    f"WHERE entity_id IN ({placeholders}) AND key = ?",
                    (*batch, program.relation),
                ).fetchall()
                found.update(
                    str(entity_id)
                    for entity_id, value, datatype in rows
                    if _compare(
                        str(value),
                        str(datatype),
                        program.comparator,
                        program.value.value,
                    )
                )
            return found
        if isinstance(program, Count):
            raise TypeError("Count produces a scalar")
        raise TypeError(type(program).__name__)

    def execute_program(self, program: Program) -> AnswerSet:
        if isinstance(program, Count):
            return AnswerSet.count(len(self._execute_entities(program.input)))
        return AnswerSet.entities(self._execute_entities(program))

    def execute_sparql(self, sparql: str) -> AnswerSet:
        marker = "# graphtask-program:"
        encoded = next(
            (
                line[len(marker) :].strip()
                for line in sparql.splitlines()
                if line.startswith(marker)
            ),
            None,
        )
        if encoded is None:
            raise ValueError("SQLite backend only executes compiled GraphTask SPARQL")
        raw = base64.urlsafe_b64decode(encoded.encode()).decode()
        return self.execute_program(parse_program(json.loads(raw)))

    def entity_info(self, entity_id: str) -> EntityInfo:
        row = self.connection.execute(
            "SELECT label, aliases_json FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return EntityInfo(entity_id=entity_id, label=entity_id)
        type_rows = self.connection.execute(
            "SELECT type_id FROM entity_types WHERE entity_id = ? ORDER BY type_id", (entity_id,)
        ).fetchall()
        return EntityInfo(
            entity_id=entity_id,
            label=str(row[0]),
            aliases=tuple(json.loads(row[1])),
            type_ids=tuple(str(value[0]) for value in type_rows),
        )

    def relation_info(self, relation_id: str) -> RelationInfo:
        row = self.connection.execute(
            "SELECT label FROM relation_labels WHERE relation_id = ?", (relation_id,)
        ).fetchone()
        return RelationInfo(relation_id=relation_id, label=str(row[0]) if row else relation_id)

    def extract_witness(self, program: Program, answers: AnswerSet) -> list[Witness]:
        graph_slice = self.materialize(program)
        return [
            Witness(answer=str(answer.value), facts=graph_slice.triples)
            for answer in answers.answers
        ]

    def materialize(
        self, program: Program, *, max_nodes: int = 10_000, max_edges: int = 50_000
    ) -> GraphSlice:
        return materialize_program(
            self,
            program,
            snapshot_id=self.snapshot_id,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def with_overlay(self, overlay: GraphOverlay) -> SQLiteGraphBackend:
        if overlay.added or overlay.removed:
            raise NotImplementedError("materialize a GraphSlice before applying overlays")
        return self

    def close(self) -> None:
        self.connection.close()
