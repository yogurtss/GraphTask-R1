from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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
        self._cache_depth = 0
        self._all_entities_cache: dict[int, tuple[str, ...]] = {}
        self._neighbors_cache: dict[
            tuple[tuple[str, ...], str, tuple[str, ...], int], tuple[Triple, ...]
        ] = {}
        self._entity_results_cache: dict[Program, frozenset[str]] = {}
        self._answer_cache: dict[Program, AnswerSet] = {}
        self._entity_info_cache: dict[str, EntityInfo] = {}
        self._relation_info_cache: dict[str, RelationInfo] = {}
        self._materialize_cache: dict[tuple[Program, int, int], GraphSlice] = {}

    @property
    def _cache_enabled(self) -> bool:
        return self._cache_depth > 0

    def _clear_query_cache(self) -> None:
        self._all_entities_cache.clear()
        self._neighbors_cache.clear()
        self._entity_results_cache.clear()
        self._answer_cache.clear()
        self._entity_info_cache.clear()
        self._relation_info_cache.clear()
        self._materialize_cache.clear()

    @contextmanager
    def query_cache(self) -> Iterator[None]:
        """Reuse immutable graph reads within one task without retaining cross-task state."""
        if self._cache_depth == 0:
            self._clear_query_cache()
        self._cache_depth += 1
        try:
            yield
        finally:
            self._cache_depth -= 1
            if self._cache_depth == 0:
                self._clear_query_cache()

    def _sql_variable_limit(self) -> int:
        runtime_limit = self.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        return max(2, min(_PREFERRED_SQL_VARIABLE_LIMIT, runtime_limit))

    def all_entities(self, *, limit: int) -> tuple[str, ...]:
        normalized_limit = max(0, limit)
        if self._cache_enabled and normalized_limit in self._all_entities_cache:
            return self._all_entities_cache[normalized_limit]
        rows = self.connection.execute(
            "SELECT entity_id FROM entities ORDER BY entity_id LIMIT ?", (normalized_limit,)
        ).fetchall()
        result = tuple(str(row[0]) for row in rows)
        if self._cache_enabled:
            self._all_entities_cache[normalized_limit] = result
        return result

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
        entities = tuple(sorted(set(entity_ids)))
        relations = tuple(sorted(set(relation_ids or ())))
        cache_key = (entities, direction, relations, limit)
        if self._cache_enabled and cache_key in self._neighbors_cache:
            return list(self._neighbors_cache[cache_key])
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
        result = tuple(sorted(found.values(), key=Triple.sort_key)[:limit])
        if self._cache_enabled:
            self._neighbors_cache[cache_key] = result
        return list(result)

    def _filter_all_entities_by_type(
        self, program: FilterType, *, max_results: int
    ) -> set[str]:
        rows = self.connection.execute(
            "SELECT typed.entity_id FROM entity_types AS typed "
            "JOIN (SELECT entity_id FROM entities ORDER BY entity_id LIMIT ?) AS candidates "
            "ON candidates.entity_id = typed.entity_id WHERE typed.type_id = ?",
            (max_results, program.type_id),
        ).fetchall()
        return {str(row[0]) for row in rows}

    def _filter_all_entities_by_literal(
        self, program: FilterLiteral, *, max_results: int
    ) -> set[str]:
        rows = self.connection.execute(
            "SELECT attribute.entity_id, attribute.value, attribute.datatype "
            "FROM attributes AS attribute "
            "JOIN (SELECT entity_id FROM entities ORDER BY entity_id LIMIT ?) AS candidates "
            "ON candidates.entity_id = attribute.entity_id WHERE attribute.key = ?",
            (max_results, program.relation),
        ).fetchall()
        return {
            str(entity_id)
            for entity_id, value, datatype in rows
            if _compare(str(value), str(datatype), program.comparator, program.value.value)
        }

    def _execute_entities(self, program: Program) -> set[str]:
        if self._cache_enabled and program in self._entity_results_cache:
            return set(self._entity_results_cache[program])
        if isinstance(program, AllEntities):
            result = set(self.all_entities(limit=program.max_results))
        elif isinstance(program, Entity):
            result = {program.entity_id}
        elif isinstance(program, Hop):
            inputs = self._execute_entities(program.input)
            edges = self.neighbors(
                sorted(inputs),
                direction=program.direction,
                relation_ids=[program.relation],
                limit=1_000_000,
            )
            result = {
                edge.object if program.direction == "out" else edge.subject for edge in edges
            }
        elif isinstance(program, Intersect):
            result = set.intersection(
                *(self._execute_entities(branch) for branch in program.inputs)
            )
        elif isinstance(program, Union):
            result = set().union(
                *(self._execute_entities(branch) for branch in program.inputs)
            )
        elif isinstance(program, FilterType):
            if isinstance(program.input, AllEntities):
                result = self._filter_all_entities_by_type(
                    program, max_results=program.input.max_results
                )
            else:
                candidates = self._execute_entities(program.input)
                result = set()
                batch_size = self._sql_variable_limit() - 1
                for batch in _chunks(sorted(candidates), batch_size):
                    placeholders = ",".join("?" for _ in batch)
                    rows = self.connection.execute(
                        f"SELECT entity_id FROM entity_types WHERE entity_id IN ({placeholders}) "
                        "AND type_id = ?",
                        (*batch, program.type_id),
                    ).fetchall()
                    result.update(str(row[0]) for row in rows)
        elif isinstance(program, FilterLiteral):
            if isinstance(program.input, AllEntities):
                result = self._filter_all_entities_by_literal(
                    program, max_results=program.input.max_results
                )
            else:
                candidates = self._execute_entities(program.input)
                result = set()
                batch_size = self._sql_variable_limit() - 1
                for batch in _chunks(sorted(candidates), batch_size):
                    placeholders = ",".join("?" for _ in batch)
                    rows = self.connection.execute(
                        f"SELECT entity_id, value, datatype FROM attributes "
                        f"WHERE entity_id IN ({placeholders}) AND key = ?",
                        (*batch, program.relation),
                    ).fetchall()
                    result.update(
                        str(entity_id)
                        for entity_id, value, datatype in rows
                        if _compare(
                            str(value),
                            str(datatype),
                            program.comparator,
                            program.value.value,
                        )
                    )
        elif isinstance(program, Count):
            raise TypeError("Count produces a scalar")
        else:
            raise TypeError(type(program).__name__)
        if self._cache_enabled:
            self._entity_results_cache[program] = frozenset(result)
        return result

    def execute_program(self, program: Program) -> AnswerSet:
        if self._cache_enabled and program in self._answer_cache:
            return self._answer_cache[program]
        if isinstance(program, Count):
            result = AnswerSet.count(len(self._execute_entities(program.input)))
        else:
            result = AnswerSet.entities(self._execute_entities(program))
        if self._cache_enabled:
            self._answer_cache[program] = result
        return result

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
        if self._cache_enabled and entity_id in self._entity_info_cache:
            return self._entity_info_cache[entity_id]
        return self.entity_infos([entity_id])[0]

    def entity_infos(self, entity_ids: Sequence[str]) -> tuple[EntityInfo, ...]:
        ordered_ids = tuple(entity_ids)
        missing = sorted(
            {
                entity_id
                for entity_id in ordered_ids
                if entity_id not in self._entity_info_cache
            }
        )
        loaded: dict[str, EntityInfo] = {}
        for batch in _chunks(missing, self._sql_variable_limit()):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                "SELECT entity.entity_id, entity.label, entity.aliases_json, typed.type_id "
                "FROM entities AS entity LEFT JOIN entity_types AS typed "
                "ON typed.entity_id = entity.entity_id "
                f"WHERE entity.entity_id IN ({placeholders}) "
                "ORDER BY entity.entity_id, typed.type_id",
                tuple(batch),
            ).fetchall()
            grouped: dict[str, tuple[str, str, list[str]]] = {}
            for raw_entity_id, label, aliases_json, type_id in rows:
                entity_id = str(raw_entity_id)
                if entity_id not in grouped:
                    grouped[entity_id] = (str(label), str(aliases_json), [])
                if type_id is not None:
                    grouped[entity_id][2].append(str(type_id))
            for entity_id in batch:
                if entity_id in grouped:
                    label, aliases_json, type_ids = grouped[entity_id]
                    loaded[entity_id] = EntityInfo(
                        entity_id=entity_id,
                        label=label,
                        aliases=tuple(json.loads(aliases_json)),
                        type_ids=tuple(type_ids),
                    )
                else:
                    loaded[entity_id] = EntityInfo(entity_id=entity_id, label=entity_id)
        if self._cache_enabled:
            self._entity_info_cache.update(loaded)
        available = {**loaded, **self._entity_info_cache}
        return tuple(available[entity_id] for entity_id in ordered_ids)

    def relation_info(self, relation_id: str) -> RelationInfo:
        if self._cache_enabled and relation_id in self._relation_info_cache:
            return self._relation_info_cache[relation_id]
        row = self.connection.execute(
            "SELECT label FROM relation_labels WHERE relation_id = ?", (relation_id,)
        ).fetchone()
        result = RelationInfo(
            relation_id=relation_id, label=str(row[0]) if row else relation_id
        )
        if self._cache_enabled:
            self._relation_info_cache[relation_id] = result
        return result

    def extract_witness(self, program: Program, answers: AnswerSet) -> list[Witness]:
        graph_slice = self.materialize(program)
        return [
            Witness(answer=str(answer.value), facts=graph_slice.triples)
            for answer in answers.answers
        ]

    def materialize(
        self, program: Program, *, max_nodes: int = 10_000, max_edges: int = 50_000
    ) -> GraphSlice:
        cache_key = (program, max_nodes, max_edges)
        if self._cache_enabled and cache_key in self._materialize_cache:
            return self._materialize_cache[cache_key]
        result = materialize_program(
            self,
            program,
            snapshot_id=self.snapshot_id,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if self._cache_enabled:
            self._materialize_cache[cache_key] = result
        return result

    def with_overlay(self, overlay: GraphOverlay) -> SQLiteGraphBackend:
        if overlay.added or overlay.removed:
            raise NotImplementedError("materialize a GraphSlice before applying overlays")
        return self

    def close(self) -> None:
        self.connection.close()
