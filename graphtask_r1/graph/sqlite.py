from __future__ import annotations

import base64
import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from graphtask_r1.graph.materialize import materialize_program
from graphtask_r1.graph.overlay import GraphOverlay
from graphtask_r1.graph.values import attribute_sort_key, format_attribute_value
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Count,
    Entity,
    EntityInfo,
    FilterLiteral,
    FilterQualifier,
    FilterType,
    GraphSlice,
    Hop,
    Intersect,
    Program,
    QueryAttribute,
    QueryAttributeQualifier,
    QueryAttributeUnderCondition,
    QueryRelation,
    QueryRelationQualifier,
    RelationInfo,
    SelectAmong,
    SelectBetween,
    Triple,
    Union,
    Verify,
    Witness,
    parse_program,
)

_PREFERRED_SQL_VARIABLE_LIMIT = 900
_LEGACY_SQL_VARIABLE_LIMIT = 999


def _time_parts(value: str | int | float) -> tuple[int, ...]:
    text = str(value).replace("-", "/")
    sign = -1 if text.startswith("/") else 1
    if sign < 0:
        text = text[1:]
    parts = tuple(int(part) for part in text.split("/"))
    return (sign * parts[0], *parts[1:])


def _chunks(values: Sequence[str], size: int) -> list[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _runtime_sql_variable_limit(connection: Any) -> int:
    """Read SQLite's bind limit, including on Python 3.10 without getlimit()."""
    getlimit = getattr(connection, "getlimit", None)
    limit_category = getattr(sqlite3, "SQLITE_LIMIT_VARIABLE_NUMBER", None)
    if callable(getlimit) and isinstance(limit_category, int):
        return int(getlimit(limit_category))
    rows = connection.execute("PRAGMA compile_options").fetchall()
    for row in rows:
        option = str(row[0])
        if option.startswith("MAX_VARIABLE_NUMBER="):
            return int(option.partition("=")[2])
    return _LEGACY_SQL_VARIABLE_LIMIT


def _compare(
    left: str,
    datatype: str,
    comparator: str,
    right: str | int | float,
    target_datatype: str,
    unit: str | None,
    target_unit: str | None,
) -> bool:
    if comparator == "contains":
        return str(right).casefold() in left.casefold()
    if target_datatype == "string" and datatype == "quantity":
        parts = str(right).split(maxsplit=1)
        try:
            right = float(parts[0])
        except ValueError:
            pass
        else:
            target_datatype = "quantity"
            target_unit = parts[1] if len(parts) == 2 else None
    elif target_datatype == "string" and datatype == "year":
        try:
            right = int(str(right))
        except ValueError:
            pass
        else:
            target_datatype = "year"
    elif target_datatype == "string" and datatype == "date":
        normalized = str(right).replace("-", "/").lstrip("/")
        if len(normalized.split("/")) == 3 and all(
            part.isdigit() for part in normalized.split("/")
        ):
            target_datatype = "date"
    if (
        datatype == "quantity"
        and target_datatype in {"quantity", "number"}
        and (unit or "1") != (target_unit or "1")
    ):
        return False
    lhs: Any
    rhs: Any
    if datatype in {"year", "date"} and target_datatype in {
        "year",
        "date",
        "number",
    }:
        left_parts = _time_parts(left)
        right_parts = _time_parts(right)
        left_year = left_parts[0]
        right_year = right_parts[0]
        if comparator in {"eq", "ne"}:
            if target_datatype in {"year", "number"}:
                equal = left_year == right_year
            else:
                equal = datatype == "date" and left_parts == right_parts
            return equal if comparator == "eq" else not equal
        if datatype == "date" and target_datatype == "date":
            lhs, rhs = left_parts, right_parts
        else:
            lhs, rhs = left_year, right_year
    elif datatype in {"quantity", "number", "year"}:
        try:
            lhs = float(left)
            rhs = float(right)
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
        self._text_search_cache: dict[tuple[str, int, int], tuple[dict[str, Any], ...]] = {}
        self._entity_resolution_cache: dict[tuple[str, str, int], tuple[str, ...]] = {}

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
        self._text_search_cache.clear()
        self._entity_resolution_cache.clear()

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
        runtime_limit = _runtime_sql_variable_limit(self.connection)
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

    def resolve_entities(
        self,
        query: str,
        *,
        match: str = "exact",
        limit: int = 5,
        trace_id: str | None = None,
    ) -> tuple[str, ...]:
        """Resolve exact IDs/titles/aliases, with optional passage-search fallback."""

        del trace_id
        normalized = query.strip()
        normalized_limit = max(0, min(limit, 20))
        if not normalized or normalized_limit == 0:
            return ()
        if match not in {"id", "exact", "search"}:
            raise ValueError(f"invalid entity match mode: {match}")
        cache_key = (normalized, match, normalized_limit)
        if self._cache_enabled and cache_key in self._entity_resolution_cache:
            return self._entity_resolution_cache[cache_key]

        result: tuple[str, ...]
        if match == "id":
            row = self.connection.execute(
                "SELECT entity_id FROM entities WHERE entity_id = ?", (normalized,)
            ).fetchone()
            result = (str(row[0]),) if row is not None else ()
        else:
            rows = self.connection.execute(
                "SELECT DISTINCT entity.entity_id FROM entities AS entity "
                "LEFT JOIN json_each(entity.aliases_json) AS alias "
                "WHERE entity.label = ? COLLATE NOCASE "
                "OR CAST(alias.value AS TEXT) = ? COLLATE NOCASE "
                "ORDER BY entity.entity_id LIMIT ?",
                (normalized, normalized, normalized_limit),
            ).fetchall()
            result = tuple(str(row[0]) for row in rows)
            if match == "search" and not result:
                try:
                    passages = self.search_text(
                        normalized,
                        limit=normalized_limit,
                        max_chars=1,
                        trace_id="resolve-entity",
                    )
                except ValueError:
                    passages = []
                result = tuple(dict.fromkeys(str(passage["page_id"]) for passage in passages))
        if self._cache_enabled:
            self._entity_resolution_cache[cache_key] = result
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

    def entity_degrees(self, entity_ids: Sequence[str]) -> dict[str, int]:
        """Return total in/out hyperlink degrees in bounded SQL batches."""

        unique_ids = tuple(sorted(set(entity_ids)))
        result = {entity_id: 0 for entity_id in unique_ids}
        batch_size = max(1, self._sql_variable_limit() // 2)
        for batch in _chunks(unique_ids, batch_size):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                "SELECT entity_id, SUM(degree) FROM ("
                f"SELECT subject AS entity_id, COUNT(*) AS degree FROM triples "
                f"WHERE subject IN ({placeholders}) GROUP BY subject UNION ALL "
                f"SELECT object AS entity_id, COUNT(*) AS degree FROM triples "
                f"WHERE object IN ({placeholders}) GROUP BY object"
                ") GROUP BY entity_id",
                (*batch, *batch),
            ).fetchall()
            for entity_id, degree in rows:
                result[str(entity_id)] = int(degree)
        return result

    def _filter_all_entities_by_type(self, program: FilterType, *, max_results: int) -> set[str]:
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
            "SELECT attribute.entity_id, attribute.value, attribute.datatype, attribute.unit "
            "FROM attributes AS attribute "
            "JOIN (SELECT entity_id FROM entities ORDER BY entity_id LIMIT ?) AS candidates "
            "ON candidates.entity_id = attribute.entity_id WHERE attribute.key = ?",
            (max_results, program.relation),
        ).fetchall()
        return {
            str(entity_id)
            for entity_id, value, datatype, unit in rows
            if _compare(
                str(value),
                str(datatype),
                program.comparator,
                program.value.value,
                program.value.datatype,
                None if unit is None else str(unit),
                program.value.unit,
            )
        }

    def _execute_hop(self, program: Hop, inputs: set[str]) -> set[str]:
        if not inputs:
            return set()
        input_column = "subject" if program.direction == "out" else "object"
        output_index = 2 if program.direction == "out" else 0
        bind_budget = self._sql_variable_limit() - 1
        entity_batch_size = max(1, bind_budget // 2)
        found: set[tuple[str, str, str]] = set()
        for entity_batch in _chunks(tuple(sorted(inputs)), entity_batch_size):
            placeholders = ",".join("?" for _ in entity_batch)
            rows = self.connection.execute(
                "SELECT subject, relation, object FROM triples "
                f"WHERE {input_column} IN ({placeholders}) AND relation = ? "
                "ORDER BY subject, relation, object LIMIT ?",
                (*entity_batch, program.relation, 1_000_000),
            ).fetchall()
            found.update((str(row[0]), str(row[1]), str(row[2])) for row in rows)
        return {row[output_index] for row in sorted(found)[:1_000_000]}

    def _source_fact_rows(self, program: Program) -> list[tuple[str, str]]:
        """Return (fact_id, projected entity) for a KoPL fact-producing operation."""

        if isinstance(program, Hop):
            inputs = self._execute_entities(program.input)
            input_column = "subject" if program.direction == "out" else "object_entity"
            output_column = "object_entity" if program.direction == "out" else "subject"
            result: list[tuple[str, str]] = []
            for batch in _chunks(sorted(inputs), self._sql_variable_limit() - 1):
                placeholders = ",".join("?" for _ in batch)
                rows = self.connection.execute(
                    f"SELECT fact_id, {output_column} FROM facts "
                    f"WHERE kind = 'relation' AND {input_column} IN ({placeholders}) "
                    "AND predicate = ? ORDER BY fact_id",
                    (*batch, program.relation),
                ).fetchall()
                result.extend((str(fact_id), str(entity_id)) for fact_id, entity_id in rows)
            return result
        if isinstance(program, FilterLiteral):
            candidates = self._execute_entities(program.input)
            result = []
            for batch in _chunks(sorted(candidates), self._sql_variable_limit() - 1):
                placeholders = ",".join("?" for _ in batch)
                rows = self.connection.execute(
                    "SELECT fact_id, subject, value, datatype, unit FROM facts "
                    f"WHERE kind = 'attribute' AND subject IN ({placeholders}) AND predicate = ? "
                    "ORDER BY fact_id",
                    (*batch, program.relation),
                ).fetchall()
                result.extend(
                    (str(fact_id), str(subject))
                    for fact_id, subject, value, datatype, unit in rows
                    if _compare(
                        str(value),
                        str(datatype),
                        program.comparator,
                        program.value.value,
                        program.value.datatype,
                        None if unit is None else str(unit),
                        program.value.unit,
                    )
                )
            return result
        raise ValueError(
            "filter_qualifier requires an immediately preceding hop or literal filter"
        )

    def _execute_filter_qualifier(self, program: FilterQualifier) -> set[str]:
        source_rows = self._source_fact_rows(program.input)
        by_fact = {fact_id: entity_id for fact_id, entity_id in source_rows}
        selected: set[str] = set()
        for batch in _chunks(sorted(by_fact), self._sql_variable_limit() - 1):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                "SELECT fact_id, value, datatype, unit FROM qualifiers "
                f"WHERE fact_id IN ({placeholders}) AND key = ? ORDER BY fact_id",
                (*batch, program.qualifier),
            ).fetchall()
            selected.update(
                by_fact[str(fact_id)]
                for fact_id, value, datatype, unit in rows
                if _compare(
                    str(value),
                    str(datatype),
                    program.comparator,
                    program.value.value,
                    program.value.datatype,
                    None if unit is None else str(unit),
                    program.value.unit,
                )
            )
        return selected

    def _execute_entities(self, program: Program) -> set[str]:
        if self._cache_enabled and program in self._entity_results_cache:
            return set(self._entity_results_cache[program])
        if isinstance(program, AllEntities):
            result = set(self.all_entities(limit=program.max_results))
        elif isinstance(program, Entity):
            result = {program.entity_id}
        elif isinstance(program, Hop):
            inputs = self._execute_entities(program.input)
            result = self._execute_hop(program, inputs)
        elif isinstance(program, Intersect):
            result = set.intersection(
                *(self._execute_entities(branch) for branch in program.inputs)
            )
        elif isinstance(program, Union):
            result = set().union(*(self._execute_entities(branch) for branch in program.inputs))
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
                        f"SELECT entity_id, value, datatype, unit FROM attributes "
                        f"WHERE entity_id IN ({placeholders}) AND key = ?",
                        (*batch, program.relation),
                    ).fetchall()
                    result.update(
                        str(entity_id)
                        for entity_id, value, datatype, unit in rows
                        if _compare(
                            str(value),
                            str(datatype),
                            program.comparator,
                            program.value.value,
                            program.value.datatype,
                            None if unit is None else str(unit),
                            program.value.unit,
                        )
                    )
        elif isinstance(program, FilterQualifier):
            result = self._execute_filter_qualifier(program)
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
        elif isinstance(
            program,
            QueryAttribute
            | QueryAttributeUnderCondition
            | QueryAttributeQualifier
            | QueryRelationQualifier,
        ):
            rows = self._literal_rows(program)
            result = AnswerSet.literals(
                {format_attribute_value(value, datatype, unit) for value, datatype, unit in rows}
            )
        elif isinstance(program, QueryRelation):
            subjects = self._execute_entities(program.subject)
            objects = self._execute_entities(program.object)
            relations: set[str] = set()
            for batch in _chunks(sorted(subjects), self._sql_variable_limit() - 1):
                placeholders = ",".join("?" for _ in batch)
                relation_rows = self.connection.execute(
                    f"SELECT relation, object FROM triples WHERE subject IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
                relations.update(
                    str(relation)
                    for relation, object_id in relation_rows
                    if str(object_id) in objects
                )
            result = AnswerSet.literals(relations)
        elif isinstance(program, Verify):
            rows = self._literal_rows(program.input)
            verified = any(
                _compare(
                    value,
                    datatype,
                    program.comparator,
                    program.value.value,
                    program.value.datatype,
                    unit,
                    program.value.unit,
                )
                for value, datatype, unit in rows
            )
            result = AnswerSet.literals(["yes" if verified else "no"])
        elif isinstance(program, SelectBetween):
            candidates = {
                *self._execute_entities(program.left),
                *self._execute_entities(program.right),
            }
            result = AnswerSet.entities(
                [self._select_by_attribute(candidates, program.attribute, program.mode)]
            )
        elif isinstance(program, SelectAmong):
            result = AnswerSet.entities(
                [
                    self._select_by_attribute(
                        self._execute_entities(program.input),
                        program.attribute,
                        program.mode,
                    )
                ]
            )
        else:
            result = AnswerSet.entities(self._execute_entities(program))
        if self._cache_enabled:
            self._answer_cache[program] = result
        return result

    def execute_entity_ids(self, program: Program) -> frozenset[str]:
        """Execute an entity-valued program without constructing an AnswerSet.

        Verification search can evaluate hundreds of internal candidates per task.  Keeping
        those candidates as IDs avoids validating an Answer object for every returned entity.
        """
        return frozenset(self._execute_entities(program))

    def discard_entity_result(self, program: Program) -> None:
        """Release a one-shot verification candidate while retaining its cached prefixes."""
        self._entity_results_cache.pop(program, None)
        self._answer_cache.pop(program, None)

    def relation_hops(
        self,
        entity_ids: Sequence[str],
        *,
        limit: int = 100,
    ) -> tuple[tuple[str, Literal["out", "in"]], ...]:
        """Return the unique relation/direction expansions from a bounded neighborhood."""
        entities = tuple(sorted(set(entity_ids)))
        if not entities or limit <= 0:
            return ()
        bind_budget = self._sql_variable_limit() - 1
        rows: set[tuple[str, str, str]] = set()
        for column in ("subject", "object"):
            for entity_batch in _chunks(entities, bind_budget):
                placeholders = ",".join("?" for _ in entity_batch)
                selected = self.connection.execute(
                    "SELECT subject, relation, object FROM triples "
                    f"WHERE {column} IN ({placeholders}) "
                    "ORDER BY subject, relation, object LIMIT ?",
                    (*entity_batch, limit),
                ).fetchall()
                rows.update((str(row[0]), str(row[1]), str(row[2])) for row in selected)
        selected_rows = sorted(rows)[:limit]
        entity_set = frozenset(entities)
        hops: dict[tuple[str, Literal["out", "in"]], None] = {}
        for subject, relation, object_id in selected_rows:
            if subject in entity_set:
                hops[(relation, "out")] = None
            if object_id in entity_set:
                hops[(relation, "in")] = None
        return tuple(hops)

    def _attribute_rows(
        self, entity_ids: set[str], attribute: str
    ) -> list[tuple[str, str, str, str | None]]:
        result: list[tuple[str, str, str, str | None]] = []
        for batch in _chunks(sorted(entity_ids), self._sql_variable_limit() - 1):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT entity_id, value, datatype, unit FROM attributes "
                f"WHERE entity_id IN ({placeholders}) AND key = ?",
                (*batch, attribute),
            ).fetchall()
            result.extend(
                (str(entity_id), str(value), str(datatype), None if unit is None else str(unit))
                for entity_id, value, datatype, unit in rows
            )
        return result

    def _attribute_fact_rows(
        self, entity_ids: set[str], attribute: str
    ) -> list[tuple[str, str, str, str, str | None]]:
        result: list[tuple[str, str, str, str, str | None]] = []
        for batch in _chunks(sorted(entity_ids), self._sql_variable_limit() - 1):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                "SELECT fact_id, subject, value, datatype, unit FROM facts "
                f"WHERE kind = 'attribute' AND subject IN ({placeholders}) AND predicate = ? "
                "ORDER BY fact_id",
                (*batch, attribute),
            ).fetchall()
            result.extend(
                (
                    str(fact_id),
                    str(entity_id),
                    str(value),
                    str(datatype),
                    None if unit is None else str(unit),
                )
                for fact_id, entity_id, value, datatype, unit in rows
            )
        return result

    def _qualifier_rows(
        self, fact_ids: Sequence[str], qualifier: str
    ) -> list[tuple[str, str, str, str | None]]:
        result: list[tuple[str, str, str, str | None]] = []
        for batch in _chunks(sorted(set(fact_ids)), self._sql_variable_limit() - 1):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                "SELECT fact_id, value, datatype, unit FROM qualifiers "
                f"WHERE fact_id IN ({placeholders}) AND key = ? ORDER BY fact_id",
                (*batch, qualifier),
            ).fetchall()
            result.extend(
                (str(fact_id), str(value), str(datatype), None if unit is None else str(unit))
                for fact_id, value, datatype, unit in rows
            )
        return result

    def _literal_rows(self, program: Program) -> list[tuple[str, str, str | None]]:
        if isinstance(program, QueryAttribute):
            return [
                (value, datatype, unit)
                for _, value, datatype, unit in self._attribute_rows(
                    self._execute_entities(program.input), program.attribute
                )
            ]
        if isinstance(program, QueryAttributeUnderCondition):
            facts = self._attribute_fact_rows(
                self._execute_entities(program.input), program.attribute
            )
            allowed = {
                fact_id
                for fact_id, value, datatype, unit in self._qualifier_rows(
                    [fact_id for fact_id, *_ in facts], program.qualifier
                )
                if _compare(
                    value,
                    datatype,
                    "eq",
                    program.qualifier_value.value,
                    program.qualifier_value.datatype,
                    unit,
                    program.qualifier_value.unit,
                )
            }
            return [
                (value, datatype, unit)
                for fact_id, _, value, datatype, unit in facts
                if fact_id in allowed
            ]
        if isinstance(program, QueryAttributeQualifier):
            facts = self._attribute_fact_rows(
                self._execute_entities(program.input), program.attribute
            )
            matching = [
                fact_id
                for fact_id, _, value, datatype, unit in facts
                if _compare(
                    value,
                    datatype,
                    "eq",
                    program.attribute_value.value,
                    program.attribute_value.datatype,
                    unit,
                    program.attribute_value.unit,
                )
            ]
            return [
                (value, datatype, unit)
                for _, value, datatype, unit in self._qualifier_rows(
                    matching, program.qualifier
                )
            ]
        if isinstance(program, QueryRelationQualifier):
            subjects = self._execute_entities(program.subject)
            objects = self._execute_entities(program.object)
            fact_ids: list[str] = []
            for batch in _chunks(sorted(subjects), self._sql_variable_limit() - 1):
                placeholders = ",".join("?" for _ in batch)
                rows = self.connection.execute(
                    "SELECT fact_id, object_entity FROM facts WHERE kind = 'relation' "
                    f"AND subject IN ({placeholders}) AND predicate = ? ORDER BY fact_id",
                    (*batch, program.relation),
                ).fetchall()
                fact_ids.extend(
                    str(fact_id)
                    for fact_id, object_id in rows
                    if str(object_id) in objects
                )
            return [
                (value, datatype, unit)
                for _, value, datatype, unit in self._qualifier_rows(
                    fact_ids, program.qualifier
                )
            ]
        if isinstance(program, QueryRelation):
            return [
                (str(value), "string", None)
                for value in self.execute_program(program).values()
            ]
        answers = self.execute_program(program)
        return [(str(value), "string", None) for value in answers.values()]

    def _select_by_attribute(self, entity_ids: set[str], attribute: str, mode: str) -> str:
        rows = self._attribute_rows(entity_ids, attribute)
        if not rows:
            raise ValueError(f"no candidate has attribute {attribute!r}")
        candidates = [
            (attribute_sort_key(value, datatype, unit), entity_id)
            for entity_id, value, datatype, unit in rows
        ]
        ordered = sorted(candidates, key=lambda item: (item[0], item[1]))
        return ordered[0][1] if mode == "min" else ordered[-1][1]

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
            {entity_id for entity_id in ordered_ids if entity_id not in self._entity_info_cache}
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
        result = RelationInfo(relation_id=relation_id, label=str(row[0]) if row else relation_id)
        if self._cache_enabled:
            self._relation_info_cache[relation_id] = result
        return result

    def all_relation_infos(self) -> tuple[RelationInfo, ...]:
        """Return the complete relation/attribute schema stored in this snapshot."""
        rows = self.connection.execute(
            "SELECT relation_id, label FROM relation_labels ORDER BY relation_id"
        ).fetchall()
        return tuple(
            RelationInfo(relation_id=str(relation_id), label=str(label))
            for relation_id, label in rows
        )

    def search_text(
        self,
        query: str,
        *,
        limit: int = 3,
        max_chars: int = 2_000,
        trace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search an optional KILT FTS5 passage sidecar without changing graph semantics."""

        del trace_id
        normalized_limit = max(0, min(limit, 100))
        normalized_max_chars = max(1, min(max_chars, 20_000))
        cache_key = (query, normalized_limit, normalized_max_chars)
        if self._cache_enabled and cache_key in self._text_search_cache:
            return [dict(value) for value in self._text_search_cache[cache_key]]
        available = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'passage_fts'"
        ).fetchone()
        if available is None:
            raise ValueError(f"snapshot {self.snapshot_id} has no text index")
        tokens = tuple(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)))
        if not tokens or normalized_limit == 0:
            return []
        expression = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        rows = self.connection.execute(
            "SELECT page_id, paragraph_id, title, substr(text, 1, ?), bm25(passage_fts) "
            "FROM passage_fts WHERE passage_fts MATCH ? "
            "ORDER BY bm25(passage_fts), page_id, paragraph_id LIMIT ?",
            (normalized_max_chars, expression, normalized_limit),
        ).fetchall()
        result = tuple(
            {
                "page_id": str(page_id),
                "paragraph_id": int(paragraph_id),
                "title": str(title),
                "text": str(text),
                "score": float(score),
            }
            for page_id, paragraph_id, title, text, score in rows
        )
        if self._cache_enabled:
            self._text_search_cache[cache_key] = result
        return [dict(value) for value in result]

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
