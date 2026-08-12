from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, cast

from graphtask_r1.dsl import canonical_signature, program_cost
from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    AllEntities,
    Count,
    Entity,
    FilterLiteral,
    FilterQualifier,
    FilterType,
    Hop,
    Intersect,
    Program,
    QueryAttribute,
    QueryAttributeQualifier,
    QueryAttributeUnderCondition,
    QueryRelation,
    QueryRelationQualifier,
    SelectAmong,
    SelectBetween,
    Union,
    Verify,
)

HopDirection = Literal["out", "in"]


@dataclass(frozen=True)
class ShortcutResult:
    found: bool | None
    program: Program | None
    explored: int
    reason: str


def topic_entities(program: Program) -> tuple[str, ...]:
    if isinstance(program, Entity):
        return (program.entity_id,)
    if isinstance(program, Intersect | Union):
        return tuple(
            sorted({entity for branch in program.inputs for entity in topic_entities(branch)})
        )
    if isinstance(
        program,
        Hop
        | FilterType
        | FilterLiteral
        | FilterQualifier
        | Count
        | QueryAttribute
        | QueryAttributeUnderCondition
        | QueryAttributeQualifier
        | Verify
        | SelectAmong,
    ):
        return topic_entities(program.input)
    if isinstance(program, QueryRelation | QueryRelationQualifier):
        return tuple(sorted({*topic_entities(program.subject), *topic_entities(program.object)}))
    if isinstance(program, SelectBetween):
        return tuple(sorted({*topic_entities(program.left), *topic_entities(program.right)}))
    if isinstance(program, AllEntities):
        return ()
    raise TypeError(type(program).__name__)


def bounded_shortcut_search(
    program: Program,
    backend: GraphBackend,
    *,
    max_candidates: int = 1000,
) -> ShortcutResult:
    gold = backend.execute_program(program)
    target_cost = program_cost(program)
    if target_cost <= 0 or not gold.answers:
        return ShortcutResult(False, None, 0, "not_applicable")
    gold_entity_ids = frozenset(gold.entity_ids())
    if len(gold_entity_ids) != len(gold.answers):
        return ShortcutResult(False, None, 0, "non_entity_answer")
    queue = deque[Program](
        Entity(entity_id=entity) for entity in topic_entities(program)
    )
    execute_entity_ids = getattr(backend, "execute_entity_ids", None)
    discard_entity_result = getattr(backend, "discard_entity_result", None)
    relation_hops = getattr(backend, "relation_hops", None)
    seen: set[str] = set()
    best_cost_by_entities: dict[frozenset[str], float] = {}
    explored = 0
    while queue:
        candidate = queue.popleft()
        signature = canonical_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        explored += 1
        if explored > max_candidates:
            return ShortcutResult(None, None, explored, "budget_exhausted")
        cost = program_cost(candidate)
        candidate_entities: frozenset[str] | None = None
        if 0 < cost < target_cost:
            if callable(execute_entity_ids):
                candidate_entities = frozenset(execute_entity_ids(candidate))
                equivalent = candidate_entities == gold_entity_ids
            else:
                candidate_answers = backend.execute_program(candidate)
                candidate_entities = frozenset(candidate_answers.entity_ids())
                equivalent = candidate_answers == gold
        else:
            equivalent = False
        if equivalent:
            return ShortcutResult(True, candidate, explored, "equivalent_lower_cost")
        if cost + 1.0 >= target_cost:
            if callable(discard_entity_result):
                discard_entity_result(candidate)
            continue
        if candidate_entities is None:
            if callable(execute_entity_ids):
                candidate_entities = frozenset(execute_entity_ids(candidate))
            else:
                candidate_entities = frozenset(backend.execute_program(candidate).entity_ids())
        values = candidate_entities
        if not values:
            continue
        best_cost = best_cost_by_entities.get(values)
        if best_cost is not None and best_cost <= cost:
            continue
        best_cost_by_entities[values] = cost
        if callable(relation_hops):
            hops = cast(
                tuple[tuple[str, HopDirection], ...], relation_hops(values, limit=100)
            )
        else:
            found: dict[tuple[str, HopDirection], None] = {}
            for triple in backend.neighbors(tuple(values), direction="both", limit=100):
                if triple.subject in values:
                    found[(triple.relation, "out")] = None
                if triple.object in values:
                    found[(triple.relation, "in")] = None
            hops = tuple(found)
        for relation, direction in hops:
            queue.append(Hop(input=candidate, relation=relation, direction=direction))
    return ShortcutResult(False, None, explored, "search_complete")
