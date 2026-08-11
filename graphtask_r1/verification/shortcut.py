from __future__ import annotations

from dataclasses import dataclass

from graphtask_r1.dsl import canonical_signature, program_cost
from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    AllEntities,
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    QueryAttribute,
    QueryRelation,
    SelectAmong,
    SelectBetween,
    Union,
)


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
    if isinstance(program, Hop | FilterType | FilterLiteral | Count | QueryAttribute | SelectAmong):
        return topic_entities(program.input)
    if isinstance(program, QueryRelation):
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
    queue: list[Program] = [Entity(entity_id=entity) for entity in topic_entities(program)]
    seen: set[str] = set()
    explored = 0
    while queue:
        candidate = queue.pop(0)
        signature = canonical_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        explored += 1
        if explored > max_candidates:
            return ShortcutResult(None, None, explored, "budget_exhausted")
        cost = program_cost(candidate)
        if 0 < cost < target_cost and backend.execute_program(candidate) == gold:
            return ShortcutResult(True, candidate, explored, "equivalent_lower_cost")
        if cost + 1.0 >= target_cost:
            continue
        values = backend.execute_program(candidate).entity_ids()
        if not values:
            continue
        for triple in backend.neighbors(values, direction="both", limit=100):
            if triple.subject in values:
                queue.append(Hop(input=candidate, relation=triple.relation, direction="out"))
            if triple.object in values:
                queue.append(Hop(input=candidate, relation=triple.relation, direction="in"))
    return ShortcutResult(False, None, explored, "search_complete")
