from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from graphtask_r1.envs.text_search import execute_text_search
from graphtask_r1.graph.base import GraphBackend
from graphtask_r1.graphscript.schema import (
    AllEntitiesOp,
    BudgetUsage,
    CountOp,
    EmitOp,
    FilterLiteralOp,
    FilterTypeOp,
    FollowOp,
    GraphScript,
    GraphScriptError,
    IntersectOp,
    PassagePagesOp,
    QueryAttributeOp,
    QueryRelationOp,
    RequireUniqueOp,
    ResolveEntityOp,
    SearchPassageOp,
    SelectAmongOp,
    SelectBetweenOp,
    StartOp,
    UnionOp,
)
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    LiteralValue,
    PassageHit,
    Program,
    QueryAttribute,
    QueryRelation,
    SelectAmong,
    SelectBetween,
    Triple,
    Union,
)


@runtime_checkable
class EntityResolverBackend(Protocol):
    def resolve_entities(
        self,
        query: str,
        *,
        match: str = "exact",
        limit: int = 5,
        trace_id: str | None = None,
    ) -> tuple[str, ...]: ...


class GraphScriptExecution(BaseModel):
    model_config = ConfigDict(frozen=True)

    program: Program
    answers: AnswerSet
    support: tuple[Triple, ...]
    relation_path: tuple[str, ...]
    usage: BudgetUsage


@dataclass(frozen=True)
class _Handle:
    program: Program | None = None
    answers: AnswerSet | None = None
    passages: tuple[PassageHit, ...] = ()


def _entity_program(entity_ids: set[str] | tuple[str, ...]) -> Program:
    values = tuple(sorted(set(entity_ids)))
    if not values:
        raise GraphScriptError("EMPTY_RESULT", "entity handle is empty")
    if len(values) == 1:
        return Entity(entity_id=values[0])
    return Union(inputs=tuple(Entity(entity_id=value) for value in values))


def _entity_ids(handle: _Handle) -> set[str]:
    if handle.answers is None:
        raise GraphScriptError("TYPE_MISMATCH", "operator requires an answer handle")
    values = set(handle.answers.entity_ids())
    if len(values) != len(handle.answers.answers):
        raise GraphScriptError("TYPE_MISMATCH", "operator requires an entity handle")
    return values


def _require_program(handle: _Handle) -> Program:
    if handle.program is None:
        raise GraphScriptError("TYPE_MISMATCH", "handle has no executable graph program")
    return handle.program


def _contains_all_entities(program: Program) -> bool:
    if isinstance(program, AllEntities):
        return True
    if isinstance(program, Intersect | Union):
        return any(_contains_all_entities(branch) for branch in program.inputs)
    if isinstance(program, Hop | FilterType | FilterLiteral | Count | QueryAttribute | SelectAmong):
        return _contains_all_entities(program.input)
    if isinstance(program, QueryRelation):
        return _contains_all_entities(program.subject) or _contains_all_entities(program.object)
    if isinstance(program, SelectBetween):
        return _contains_all_entities(program.left) or _contains_all_entities(program.right)
    return False


def _resolve(
    backend: GraphBackend,
    op: ResolveEntityOp,
    *,
    trace_id: str | None,
) -> tuple[str, ...]:
    if not isinstance(backend, EntityResolverBackend):
        raise GraphScriptError(
            "ENTITY_RESOLUTION_UNAVAILABLE",
            "current graph backend does not implement bounded entity resolution",
        )
    values = backend.resolve_entities(
        op.query,
        match=op.match,
        limit=op.limit,
        trace_id=trace_id,
    )
    if not values:
        raise GraphScriptError("ENTITY_NOT_FOUND", f"cannot resolve entity: {op.query!r}")
    return values


def graphscript_to_program(
    script: GraphScript,
    *,
    seed_entity: str | None = None,
    backend: GraphBackend | None = None,
) -> Program:
    """Compile v0.1 statically and v0.2 through the same bounded runtime semantics."""

    if script.version == "0.1":
        if seed_entity is None:
            raise GraphScriptError("MISSING_SEED", "v0.1 requires a topic seed")
        handles: dict[str, Program] = {}
        emitted: Program | None = None
        for op in script.ops:
            if isinstance(op, StartOp):
                handles[op.out] = Entity(entity_id=seed_entity)
            elif isinstance(op, FollowOp):
                handles[op.out] = Hop(
                    input=handles[op.input_handle],
                    relation=op.relation,
                    direction=op.direction,
                )
            elif isinstance(op, EmitOp):
                emitted = handles[op.input_handle]
        if emitted is None:
            raise GraphScriptError("MISSING_EMIT", "script does not emit a handle")
        return emitted
    if backend is None:
        raise GraphScriptError(
            "MISSING_BACKEND", "v0.2 resolution requires the execution graph backend"
        )
    relations = frozenset(op.relation for op in script.ops if isinstance(op, FollowOp))
    return execute_graphscript(
        script,
        backend,
        seed_entity=seed_entity,
        allowed_relations=relations,
        max_edge_visits=1_000_000,
        max_returned_entities=100_000,
    ).program


def program_to_graphscript(
    program: Program,
    *,
    follow_limit: int = 100,
    version: str = "0.1",
    entity_reference: Callable[[str], tuple[str, Literal["id", "exact", "search"]]] | None = None,
) -> GraphScript:
    if version == "0.1":
        if not isinstance(program, Hop) or not isinstance(program.input, Hop):
            raise GraphScriptError("INVALID_SHAPE", "program must contain exactly two chained hops")
        first = program.input
        if not isinstance(first.input, Entity):
            raise GraphScriptError("INVALID_SHAPE", "program must start from one entity")
        return GraphScript.model_validate(
            {
                "version": "0.1",
                "ops": [
                    {"op": "start", "entity": "$seed", "out": "h0"},
                    {
                        "op": "follow",
                        "in": "h0",
                        "relation": first.relation,
                        "direction": first.direction,
                        "limit": follow_limit,
                        "out": "h1",
                    },
                    {
                        "op": "follow",
                        "in": "h1",
                        "relation": program.relation,
                        "direction": program.direction,
                        "limit": follow_limit,
                        "out": "h2",
                    },
                    {"op": "require_unique", "in": "h2"},
                    {"op": "emit", "in": "h2"},
                ],
            }
        )
    if version != "0.2":
        raise GraphScriptError("UNSUPPORTED_VERSION", f"unsupported version: {version}")

    ops: list[dict[str, object]] = []
    next_handle = 0

    def allocate() -> str:
        nonlocal next_handle
        if next_handle > 63:
            raise GraphScriptError("PROGRAM_TOO_LARGE", "v0.2 supports at most 64 handles")
        handle = f"h{next_handle}"
        next_handle += 1
        return handle

    def compile_node(node: Program) -> str:
        if isinstance(node, AllEntities):
            output = allocate()
            ops.append(
                {
                    "op": "all_entities",
                    "max_results": node.max_results,
                    "out": output,
                }
            )
            return output
        if isinstance(node, Entity):
            output = allocate()
            query, match = (
                entity_reference(node.entity_id)
                if entity_reference is not None
                else (node.entity_id, "id")
            )
            ops.append(
                {
                    "op": "resolve_entity",
                    "query": query,
                    "match": match,
                    "limit": 1,
                    "out": output,
                }
            )
            return output
        if isinstance(node, Hop):
            source = compile_node(node.input)
            output = allocate()
            ops.append(
                {
                    "op": "follow",
                    "in": source,
                    "relation": node.relation,
                    "direction": node.direction,
                    "limit": follow_limit,
                    "out": output,
                }
            )
            return output
        if isinstance(node, Intersect | Union):
            inputs = [compile_node(branch) for branch in node.inputs]
            output = allocate()
            ops.append(
                {
                    "op": "intersect" if isinstance(node, Intersect) else "union",
                    "inputs": inputs,
                    "out": output,
                }
            )
            return output
        if isinstance(node, FilterType):
            source = compile_node(node.input)
            output = allocate()
            ops.append({"op": "filter_type", "in": source, "type_id": node.type_id, "out": output})
            return output
        if isinstance(node, FilterLiteral):
            source = compile_node(node.input)
            output = allocate()
            ops.append(
                {
                    "op": "filter_literal",
                    "in": source,
                    "relation": node.relation,
                    "comparator": node.comparator,
                    "value": node.value.model_dump(mode="json"),
                    "out": output,
                }
            )
            return output
        if isinstance(node, Count):
            source = compile_node(node.input)
            output = allocate()
            ops.append({"op": "count", "in": source, "out": output})
            return output
        if isinstance(node, QueryAttribute):
            source = compile_node(node.input)
            output = allocate()
            ops.append(
                {
                    "op": "query_attribute",
                    "in": source,
                    "attribute": node.attribute,
                    "out": output,
                }
            )
            return output
        if isinstance(node, QueryRelation):
            subject = compile_node(node.subject)
            object_ = compile_node(node.object)
            output = allocate()
            ops.append(
                {
                    "op": "query_relation",
                    "subject": subject,
                    "object": object_,
                    "out": output,
                }
            )
            return output
        if isinstance(node, SelectBetween):
            left = compile_node(node.left)
            right = compile_node(node.right)
            output = allocate()
            ops.append(
                {
                    "op": "select_between",
                    "left": left,
                    "right": right,
                    "attribute": node.attribute,
                    "mode": node.mode,
                    "out": output,
                }
            )
            return output
        if isinstance(node, SelectAmong):
            source = compile_node(node.input)
            output = allocate()
            ops.append(
                {
                    "op": "select_among",
                    "in": source,
                    "attribute": node.attribute,
                    "mode": node.mode,
                    "out": output,
                }
            )
            return output
        raise GraphScriptError(
            "UNSUPPORTED_PROGRAM", f"cannot convert {type(node).__name__} to v0.2"
        )

    final_handle = compile_node(program)
    ops.append({"op": "emit", "in": final_handle})
    return GraphScript.model_validate({"version": "0.2", "ops": ops})


def execute_graphscript(
    script: GraphScript,
    backend: GraphBackend,
    *,
    seed_entity: str | None = None,
    allowed_relations: frozenset[str],
    max_edge_visits: int,
    max_returned_entities: int = 1_000,
    trace_id: str | None = None,
) -> GraphScriptExecution:
    handles: dict[str, _Handle] = {}
    support: set[Triple] = set()
    relation_path: list[str] = []
    edge_visits = 0
    returned_entities = 0
    graph_calls = 0
    passage_searches = 0
    returned_passages = 0
    emitted: _Handle | None = None
    program: Program
    answers: AnswerSet
    values: tuple[str, ...] | set[str]

    def account_answers(answers: AnswerSet) -> None:
        nonlocal returned_entities
        returned_entities += len(answers.entity_ids())
        if returned_entities > max_returned_entities:
            raise GraphScriptError("BUDGET_EXCEEDED", "returned-entity budget exceeded")

    def execute_program(program: Program) -> AnswerSet:
        nonlocal edge_visits, graph_calls
        try:
            answers = backend.execute_program(program)
            witnesses = (
                () if _contains_all_entities(program) else backend.extract_witness(program, answers)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GraphScriptError("EXECUTION_ERROR", str(exc)) from exc
        graph_calls += 1
        facts = {fact for witness in witnesses for fact in witness.facts}
        additions = facts - support
        if edge_visits + len(additions) > max_edge_visits:
            raise GraphScriptError("BUDGET_EXCEEDED", "edge-visit budget exceeded")
        support.update(additions)
        edge_visits += len(additions)
        account_answers(answers)
        return answers

    for index, op in enumerate(script.ops):
        op_trace = f"{trace_id or 'graphscript'}:{index}"
        if isinstance(op, StartOp):
            if seed_entity is None:
                raise GraphScriptError("MISSING_SEED", "start($seed) requires a topic entity")
            program = Entity(entity_id=seed_entity)
            answers = AnswerSet.entities([seed_entity])
            account_answers(answers)
            handles[op.out] = _Handle(program=program, answers=answers)
        elif isinstance(op, AllEntitiesOp):
            handles[op.out] = _Handle(program=AllEntities(max_results=op.max_results))
        elif isinstance(op, ResolveEntityOp):
            values = _resolve(backend, op, trace_id=op_trace)
            program = _entity_program(values)
            answers = AnswerSet.entities(values)
            account_answers(answers)
            handles[op.out] = _Handle(program=program, answers=answers)
        elif isinstance(op, SearchPassageOp):
            try:
                passages = execute_text_search(
                    backend,
                    op.query,
                    limit=op.limit,
                    max_chars=op.max_chars,
                    trace_id=op_trace,
                )
            except ValueError as exc:
                raise GraphScriptError("PASSAGE_SEARCH_ERROR", str(exc)) from exc
            if not passages:
                raise GraphScriptError("EMPTY_RESULT", "passage search returned no results")
            passage_searches += 1
            returned_passages += len(passages)
            handles[op.out] = _Handle(passages=passages)
        elif isinstance(op, PassagePagesOp):
            pages = tuple(passage.page_id for passage in handles[op.input_handle].passages)
            program = _entity_program(pages)
            answers = AnswerSet.entities(pages)
            account_answers(answers)
            handles[op.out] = _Handle(program=program, answers=answers)
        elif isinstance(op, FollowOp):
            if op.relation not in allowed_relations:
                raise GraphScriptError(
                    "RELATION_NOT_ALLOWED",
                    f"relation is not in the episode catalog: {op.relation}",
                )
            remaining = max_edge_visits - edge_visits
            if remaining <= 0:
                raise GraphScriptError("BUDGET_EXCEEDED", "edge-visit budget exhausted")
            source = handles[op.input_handle]
            triples = backend.neighbors(
                sorted(_entity_ids(source)),
                direction=op.direction,
                relation_ids=[op.relation],
                limit=min(op.limit, remaining + 1),
                trace_id=op_trace,
            )
            graph_calls += 1
            edge_visits += len(triples)
            if edge_visits > max_edge_visits:
                raise GraphScriptError("BUDGET_EXCEEDED", "edge-visit budget exceeded")
            values = {
                triple.object if op.direction == "out" else triple.subject for triple in triples
            }
            answers = AnswerSet.entities(values)
            account_answers(answers)
            handles[op.out] = _Handle(
                program=Hop(
                    input=_require_program(source),
                    relation=op.relation,
                    direction=op.direction,
                ),
                answers=answers,
            )
            support.update(triples)
            relation_path.append(op.relation)
        elif isinstance(op, IntersectOp | UnionOp):
            branches = tuple(handles[value] for value in op.inputs)
            entity_sets = [_entity_ids(branch) for branch in branches]
            values = (
                set.intersection(*entity_sets)
                if isinstance(op, IntersectOp)
                else set().union(*entity_sets)
            )
            program_type = Intersect if isinstance(op, IntersectOp) else Union
            program = program_type(inputs=tuple(_require_program(branch) for branch in branches))
            answers = AnswerSet.entities(values)
            account_answers(answers)
            handles[op.out] = _Handle(program=program, answers=answers)
        elif isinstance(op, FilterTypeOp):
            source = handles[op.input_handle]
            program = FilterType(input=_require_program(source), type_id=op.type_id)
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, FilterLiteralOp):
            source = handles[op.input_handle]
            program = FilterLiteral(
                input=_require_program(source),
                relation=op.relation,
                comparator=op.comparator,
                value=LiteralValue.model_validate(op.value.model_dump()),
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, CountOp):
            source = handles[op.input_handle]
            program = Count(input=_require_program(source))
            answers = (
                AnswerSet.count(len(_entity_ids(source)))
                if source.answers is not None
                else execute_program(program)
            )
            handles[op.out] = _Handle(program=program, answers=answers)
        elif isinstance(op, QueryAttributeOp):
            source = handles[op.input_handle]
            program = QueryAttribute(input=_require_program(source), attribute=op.attribute)
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, QueryRelationOp):
            subject = handles[op.subject]
            object_ = handles[op.object]
            program = QueryRelation(
                subject=_require_program(subject), object=_require_program(object_)
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, SelectBetweenOp):
            left = handles[op.left]
            right = handles[op.right]
            program = SelectBetween(
                left=_require_program(left),
                right=_require_program(right),
                attribute=op.attribute,
                mode=op.mode,
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, SelectAmongOp):
            source = handles[op.input_handle]
            program = SelectAmong(
                input=_require_program(source), attribute=op.attribute, mode=op.mode
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, RequireUniqueOp):
            unique_answers = handles[op.input_handle].answers
            if unique_answers is None or not unique_answers.answers:
                raise GraphScriptError("EMPTY_RESULT", "final handle is empty")
            if len(unique_answers.answers) != 1:
                raise GraphScriptError(
                    "NON_UNIQUE_RESULT",
                    f"final handle contains {len(unique_answers.answers)} answers",
                )
        elif isinstance(op, EmitOp):
            emitted = handles[op.input_handle]

    if emitted is None or emitted.answers is None or emitted.program is None:
        raise GraphScriptError("MISSING_EMIT", "script did not emit an executable answer")
    if not emitted.answers.answers:
        raise GraphScriptError("EMPTY_RESULT", "emitted answer is empty")
    return GraphScriptExecution(
        program=emitted.program,
        answers=emitted.answers,
        support=tuple(sorted(support, key=Triple.sort_key)),
        relation_path=tuple(relation_path),
        usage=BudgetUsage(
            edge_visits=edge_visits,
            operators=len(script.ops),
            returned_entities=returned_entities,
            graph_calls=graph_calls,
            passage_searches=passage_searches,
            returned_passages=returned_passages,
        ),
    )
