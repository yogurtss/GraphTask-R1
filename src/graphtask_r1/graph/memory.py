from __future__ import annotations

import base64
import json
from collections.abc import Sequence

from graphtask_r1.graph.overlay import GraphOverlay
from graphtask_r1.schema import (
    AnswerSet,
    Count,
    Entity,
    EntityInfo,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    RelationInfo,
    Triple,
    Witness,
    parse_program,
)


def _literal(value: str) -> str | int | float:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _compare(left: str | int | float, op: str, right: str | int | float) -> bool:
    if op == "contains":
        return str(right).casefold() in str(left).casefold()
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    try:
        lhs_number, rhs_number = float(left), float(right)
    except (TypeError, ValueError):
        lhs_text, rhs_text = str(left), str(right)
        return {
            "lt": lhs_text < rhs_text,
            "le": lhs_text <= rhs_text,
            "gt": lhs_text > rhs_text,
            "ge": lhs_text >= rhs_text,
        }[op]
    return {
        "lt": lhs_number < rhs_number,
        "le": lhs_number <= rhs_number,
        "gt": lhs_number > rhs_number,
        "ge": lhs_number >= rhs_number,
    }[op]


class InMemoryGraphBackend:
    """Immutable, deterministic graph backend used by tests and offline pipelines."""

    def __init__(
        self,
        triples: Sequence[Triple],
        entities: Sequence[EntityInfo] = (),
        relations: Sequence[RelationInfo] = (),
    ) -> None:
        self._triples = tuple(sorted(set(triples), key=Triple.sort_key))
        self._entities = {entity.entity_id: entity for entity in entities}
        self._relations = {relation.relation_id: relation for relation in relations}

    @property
    def triples(self) -> tuple[Triple, ...]:
        return self._triples

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
        entity_set = set(entity_ids)
        relation_set = set(relation_ids) if relation_ids else None
        selected = [
            triple
            for triple in self._triples
            if (relation_set is None or triple.relation in relation_set)
            and (
                (direction in {"out", "both"} and triple.subject in entity_set)
                or (direction in {"in", "both"} and triple.object in entity_set)
            )
        ]
        return selected[: max(0, limit)]

    def _execute_entities(self, program: Program) -> set[str]:
        if isinstance(program, Entity):
            return {program.entity_id}
        if isinstance(program, Hop):
            inputs = self._execute_entities(program.input)
            triples = self.neighbors(
                sorted(inputs), direction=program.direction, relation_ids=[program.relation]
            )
            if program.direction == "out":
                return {triple.object for triple in triples}
            return {triple.subject for triple in triples}
        if isinstance(program, Intersect):
            branches = [self._execute_entities(branch) for branch in program.inputs]
            return set.intersection(*branches)
        if isinstance(program, FilterType):
            return {
                entity_id
                for entity_id in self._execute_entities(program.input)
                if program.type_id in self.entity_info(entity_id).type_ids
            }
        if isinstance(program, FilterLiteral):
            result: set[str] = set()
            for entity_id in self._execute_entities(program.input):
                values = self.neighbors(
                    [entity_id], direction="out", relation_ids=[program.relation]
                )
                if any(
                    _compare(_literal(triple.object), program.comparator, program.value)
                    for triple in values
                ):
                    result.add(entity_id)
            return result
        if isinstance(program, Count):
            raise TypeError("Count produces a scalar, not an entity set")
        raise TypeError(f"unsupported program type: {type(program).__name__}")

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
            raise ValueError("SPARQL is missing the executable GraphTask program marker")
        raw = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        return self.execute_program(parse_program(json.loads(raw)))

    def entity_info(self, entity_id: str) -> EntityInfo:
        return self._entities.get(entity_id, EntityInfo(entity_id=entity_id, label=entity_id))

    def relation_info(self, relation_id: str) -> RelationInfo:
        return self._relations.get(
            relation_id, RelationInfo(relation_id=relation_id, label=relation_id.replace("_", " "))
        )

    def extract_witness(self, program: Program, answers: AnswerSet) -> list[Witness]:
        facts = self._program_facts(program)
        return [Witness(answer=str(answer.value), facts=facts) for answer in answers.answers]

    def _program_facts(self, program: Program) -> tuple[Triple, ...]:
        if isinstance(program, Entity):
            return ()
        if isinstance(program, FilterType | Count):
            return self._program_facts(program.input)
        if isinstance(program, FilterLiteral):
            inputs = self._execute_entities(program.input)
            own = self.neighbors(sorted(inputs), direction="out", relation_ids=[program.relation])
            return tuple(
                sorted(set(self._program_facts(program.input)) | set(own), key=Triple.sort_key)
            )
        if isinstance(program, Hop):
            inputs = self._execute_entities(program.input)
            own = self.neighbors(
                sorted(inputs), direction=program.direction, relation_ids=[program.relation]
            )
            return tuple(
                sorted(set(self._program_facts(program.input)) | set(own), key=Triple.sort_key)
            )
        if isinstance(program, Intersect):
            facts = {fact for branch in program.inputs for fact in self._program_facts(branch)}
            return tuple(sorted(facts, key=Triple.sort_key))
        return ()

    def with_overlay(self, overlay: GraphOverlay) -> InMemoryGraphBackend:
        return InMemoryGraphBackend(
            overlay.apply(self._triples),
            tuple(self._entities.values()),
            tuple(self._relations.values()),
        )


def toy_graph() -> InMemoryGraphBackend:
    entities = [
        EntityInfo(entity_id="alice", label="Alice", type_ids=("person",)),
        EntityInfo(entity_id="bob", label="Bob", type_ids=("person",)),
        EntityInfo(entity_id="cara", label="Cara", type_ids=("person",)),
        EntityInfo(entity_id="acme", label="Acme", type_ids=("company",)),
        EntityInfo(entity_id="globex", label="Globex", type_ids=("company",)),
        EntityInfo(entity_id="paris", label="Paris", type_ids=("city",)),
        EntityInfo(entity_id="london", label="London", type_ids=("city",)),
        EntityInfo(entity_id="france", label="France", type_ids=("country",)),
        EntityInfo(entity_id="uk", label="United Kingdom", aliases=("UK",), type_ids=("country",)),
    ]
    triples = [
        Triple(subject="alice", relation="works_at", object="acme"),
        Triple(subject="bob", relation="works_at", object="acme"),
        Triple(subject="cara", relation="works_at", object="globex"),
        Triple(subject="alice", relation="friend", object="bob"),
        Triple(subject="bob", relation="friend", object="cara"),
        Triple(subject="alice", relation="friend_of_friend", object="cara"),
        Triple(subject="acme", relation="located_in", object="paris"),
        Triple(subject="globex", relation="located_in", object="london"),
        Triple(subject="paris", relation="country", object="france"),
        Triple(subject="london", relation="country", object="uk"),
        Triple(subject="alice", relation="age", object="34"),
        Triple(subject="bob", relation="age", object="29"),
        Triple(subject="cara", relation="age", object="41"),
    ]
    return InMemoryGraphBackend(triples, entities)
