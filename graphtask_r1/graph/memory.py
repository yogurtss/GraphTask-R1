from __future__ import annotations

import base64
import json
from collections.abc import Sequence

from graphtask_r1.graph.overlay import GraphOverlay
from graphtask_r1.graph.values import attribute_sort_key
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
    QueryAttribute,
    QueryRelation,
    RelationInfo,
    SelectAmong,
    SelectBetween,
    Triple,
    Union,
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

    def all_entities(self, *, limit: int) -> tuple[str, ...]:
        values = set(self._entities)
        if not values:
            values.update(triple.subject for triple in self._triples)
            values.update(triple.object for triple in self._triples)
        return tuple(sorted(values)[: max(0, limit)])

    def resolve_entities(
        self,
        query: str,
        *,
        match: str = "exact",
        limit: int = 5,
        trace_id: str | None = None,
    ) -> tuple[str, ...]:
        """Resolve an ID, label, or alias without relying on global search state."""

        del trace_id
        normalized = query.strip().casefold()
        if not normalized or limit <= 0:
            return ()
        if match == "id":
            return (query,) if query in self._entities else ()
        exact = [
            entity_id
            for entity_id, entity in self._entities.items()
            if normalized
            in {entity.label.casefold(), *(alias.casefold() for alias in entity.aliases)}
        ]
        if match == "exact" or exact:
            return tuple(sorted(exact)[:limit])
        if match != "search":
            raise ValueError(f"invalid entity match mode: {match}")
        fuzzy = [
            entity_id
            for entity_id, entity in self._entities.items()
            if normalized in entity.label.casefold()
            or any(normalized in alias.casefold() for alias in entity.aliases)
        ]
        return tuple(sorted(fuzzy)[:limit])

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
        if isinstance(program, AllEntities):
            return set(self.all_entities(limit=program.max_results))
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
        if isinstance(program, Union):
            return set().union(*(self._execute_entities(branch) for branch in program.inputs))
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
                    _compare(_literal(triple.object), program.comparator, program.value.value)
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
        if isinstance(program, QueryAttribute):
            values = {
                triple.object
                for triple in self._triples
                if triple.subject in self._execute_entities(program.input)
                and triple.relation == program.attribute
            }
            return AnswerSet.literals(values)
        if isinstance(program, QueryRelation):
            subjects = self._execute_entities(program.subject)
            objects = self._execute_entities(program.object)
            return AnswerSet.literals(
                {
                    triple.relation
                    for triple in self._triples
                    if triple.subject in subjects and triple.object in objects
                }
            )
        if isinstance(program, SelectBetween):
            candidates = {
                *self._execute_entities(program.left),
                *self._execute_entities(program.right),
            }
            return AnswerSet.entities(
                [self._select_by_attribute(candidates, program.attribute, program.mode)]
            )
        if isinstance(program, SelectAmong):
            return AnswerSet.entities(
                [
                    self._select_by_attribute(
                        self._execute_entities(program.input),
                        program.attribute,
                        program.mode,
                    )
                ]
            )
        return AnswerSet.entities(self._execute_entities(program))

    def _select_by_attribute(self, entity_ids: set[str], attribute: str, mode: str) -> str:
        candidates: list[tuple[tuple[int, str, float, int, int], str]] = []
        for triple in self._triples:
            if triple.subject not in entity_ids or triple.relation != attribute:
                continue
            value = _literal(triple.object)
            datatype = "quantity" if isinstance(value, int | float) else "string"
            candidates.append(
                (attribute_sort_key(str(triple.object), datatype, None), triple.subject)
            )
        if not candidates:
            raise ValueError(f"no candidate has attribute {attribute!r}")
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
            raise ValueError("SPARQL is missing the executable GraphTask program marker")
        raw = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        return self.execute_program(parse_program(json.loads(raw)))

    def entity_info(self, entity_id: str) -> EntityInfo:
        return self._entities.get(entity_id, EntityInfo(entity_id=entity_id, label=entity_id))

    def relation_info(self, relation_id: str) -> RelationInfo:
        return self._relations.get(
            relation_id, RelationInfo(relation_id=relation_id, label=relation_id.replace("_", " "))
        )

    def all_relation_infos(self) -> tuple[RelationInfo, ...]:
        relation_ids = sorted(
            {*self._relations, *(triple.relation for triple in self._triples)}
        )
        return tuple(self.relation_info(relation_id) for relation_id in relation_ids)

    def extract_witness(self, program: Program, answers: AnswerSet) -> list[Witness]:
        facts = self._program_facts(program)
        return [Witness(answer=str(answer.value), facts=facts) for answer in answers.answers]

    def _program_facts(self, program: Program) -> tuple[Triple, ...]:
        if isinstance(program, Entity | AllEntities):
            return ()
        if isinstance(program, FilterType | Count):
            return self._program_facts(program.input)
        if isinstance(program, QueryAttribute | SelectAmong):
            inputs = self._execute_entities(program.input)
            own = self.neighbors(sorted(inputs), direction="out", relation_ids=[program.attribute])
            return tuple(
                sorted(set(self._program_facts(program.input)) | set(own), key=Triple.sort_key)
            )
        if isinstance(program, QueryRelation):
            subjects = self._execute_entities(program.subject)
            objects = self._execute_entities(program.object)
            relation_facts = {
                triple
                for triple in self._triples
                if triple.subject in subjects and triple.object in objects
            }
            inherited = set(self._program_facts(program.subject)) | set(
                self._program_facts(program.object)
            )
            return tuple(sorted(inherited | relation_facts, key=Triple.sort_key))
        if isinstance(program, SelectBetween):
            candidates = {
                *self._execute_entities(program.left),
                *self._execute_entities(program.right),
            }
            own = self.neighbors(
                sorted(candidates), direction="out", relation_ids=[program.attribute]
            )
            inherited = set(self._program_facts(program.left)) | set(
                self._program_facts(program.right)
            )
            return tuple(sorted(inherited | set(own), key=Triple.sort_key))
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
        if isinstance(program, Intersect | Union):
            facts = {fact for branch in program.inputs for fact in self._program_facts(branch)}
            return tuple(sorted(facts, key=Triple.sort_key))
        return ()

    def materialize(
        self, program: Program, *, max_nodes: int = 10_000, max_edges: int = 50_000
    ) -> GraphSlice:
        answers = self.execute_program(program)
        facts = self._program_facts(program)
        node_ids = sorted(
            {value for fact in facts for value in (fact.subject, fact.object)}
            | set(answers.entity_ids())
        )
        truncated = len(facts) > max_edges or len(node_ids) > max_nodes
        selected_facts = facts[:max_edges]
        selected_ids = set(node_ids[:max_nodes])
        return GraphSlice(
            snapshot_id="memory",
            triples=selected_facts,
            entities=tuple(self.entity_info(entity_id) for entity_id in sorted(selected_ids)),
            relations=tuple(
                self.relation_info(relation_id)
                for relation_id in sorted({fact.relation for fact in selected_facts})
            ),
            complete=not truncated,
            truncated=truncated,
            remote_answers=answers,
        )

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
