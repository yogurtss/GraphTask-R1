from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
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
    FilterQualifierOp,
    FilterTypeOp,
    FollowOp,
    GraphScript,
    GraphScriptError,
    IntersectOp,
    PassagePagesOp,
    QueryAttributeOp,
    QueryAttributeQualifierOp,
    QueryAttributeUnderConditionOp,
    QueryRelationOp,
    QueryRelationQualifierOp,
    RequireUniqueOp,
    ResolveEntityOp,
    SearchPassageOp,
    SelectAmongOp,
    SelectBetweenOp,
    StartOp,
    UnionOp,
    VerifyOp,
    graphscript_operators,
)
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Count,
    Entity,
    FilterLiteral,
    FilterQualifier,
    FilterType,
    Hop,
    Intersect,
    LiteralValue,
    PassageHit,
    Program,
    QueryAttribute,
    QueryAttributeQualifier,
    QueryAttributeUnderCondition,
    QueryRelation,
    QueryRelationQualifier,
    SelectAmong,
    SelectBetween,
    Triple,
    Union,
    Verify,
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
    steps: tuple[ExecutionStepTrace, ...] = ()


class HandleTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["entity", "answer", "passage", "program", "empty"]
    state: Literal["materialized", "deferred", "empty"] = "materialized"
    values: tuple[str, ...] = ()
    total_count: int = 0
    truncated: bool = False
    limit: int | None = None


class ExecutionStepTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    operation: dict[str, object]
    input_handles: dict[str, HandleTrace]
    output_handle: str | None = None
    output: HandleTrace | None = None
    selected_entities: tuple[str, ...] = ()
    retrieved_entities: tuple[str, ...] = ()
    discarded_entities: tuple[str, ...] = ()
    new_evidence: tuple[Triple, ...] = ()
    new_evidence_total: int = 0
    evidence_truncated: bool = False
    latency_ms: float = 0.0
    cumulative_usage: BudgetUsage


@dataclass(frozen=True)
class _Handle:
    program: Program | None = None
    answers: AnswerSet | None = None
    passages: tuple[PassageHit, ...] = ()


def _input_handle_names(op: object) -> tuple[str, ...]:
    if isinstance(op, IntersectOp | UnionOp):
        return op.inputs
    if isinstance(op, QueryRelationOp | QueryRelationQualifierOp):
        return (op.subject, op.object)
    if isinstance(op, SelectBetweenOp):
        return (op.left, op.right)
    if isinstance(
        op,
        PassagePagesOp
        | FollowOp
        | FilterTypeOp
        | FilterLiteralOp
        | FilterQualifierOp
        | CountOp
        | QueryAttributeOp
        | QueryAttributeUnderConditionOp
        | QueryAttributeQualifierOp
        | VerifyOp
        | SelectAmongOp
        | RequireUniqueOp
        | EmitOp,
    ):
        return (op.input_handle,)
    return ()


def _preferred_values(
    values: tuple[str, ...], preferred: set[str], limit: int
) -> tuple[str, ...]:
    value_set = set(values)
    selected = [value for value in sorted(preferred) if value in value_set]
    selected_set = set(selected)
    selected.extend(value for value in values if value not in selected_set)
    return tuple(selected[:limit])


def _handle_trace(
    handle: _Handle, *, preview_limit: int, preferred: set[str] | None = None
) -> HandleTrace:
    preferred_values = preferred or set()
    if handle.answers is not None:
        raw_values = tuple(str(answer.value) for answer in handle.answers.answers)
        entity_values = handle.answers.entity_ids()
        kind: Literal["entity", "answer", "passage", "program", "empty"] = (
            "entity" if len(entity_values) == len(raw_values) else "answer"
        )
        return HandleTrace(
            kind=kind,
            values=_preferred_values(raw_values, preferred_values, preview_limit),
            total_count=len(raw_values),
            truncated=len(raw_values) > preview_limit,
        )
    if handle.passages:
        values = tuple(passage.page_id for passage in handle.passages)
        return HandleTrace(
            kind="passage",
            values=_preferred_values(values, preferred_values, preview_limit),
            total_count=len(values),
            truncated=len(values) > preview_limit,
        )
    if handle.program is not None:
        if isinstance(handle.program, AllEntities):
            return HandleTrace(
                kind="program",
                state="deferred",
                limit=handle.program.max_results,
            )
        return HandleTrace(kind="program", state="deferred")
    return HandleTrace(kind="empty", state="empty")


def _handle_entity_ids(handle: _Handle) -> set[str]:
    if handle.answers is None:
        return set()
    return set(handle.answers.entity_ids())


def _bounded(
    values: set[str], limit: int, *, preferred: set[str] | None = None
) -> tuple[str, ...]:
    return _preferred_values(tuple(sorted(values)), preferred or set(), limit)


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
        return _contains_all_entities(program.input)
    if isinstance(program, QueryRelation | QueryRelationQualifier):
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
    """Compile v0.1 statically and v0.2/v0.3 with bounded runtime semantics."""

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
    if version not in {"0.2", "0.3"}:
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
        if isinstance(node, FilterQualifier):
            source = compile_node(node.input)
            output = allocate()
            ops.append(
                {
                    "op": "filter_qualifier",
                    "in": source,
                    "qualifier": node.qualifier,
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
        if isinstance(node, QueryAttributeUnderCondition):
            source = compile_node(node.input)
            output = allocate()
            ops.append(
                {
                    "op": "query_attribute_under_condition",
                    "in": source,
                    "attribute": node.attribute,
                    "qualifier": node.qualifier,
                    "qualifier_value": node.qualifier_value.model_dump(mode="json"),
                    "out": output,
                }
            )
            return output
        if isinstance(node, QueryAttributeQualifier):
            source = compile_node(node.input)
            output = allocate()
            ops.append(
                {
                    "op": "query_attribute_qualifier",
                    "in": source,
                    "attribute": node.attribute,
                    "attribute_value": node.attribute_value.model_dump(mode="json"),
                    "qualifier": node.qualifier,
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
        if isinstance(node, QueryRelationQualifier):
            subject = compile_node(node.subject)
            object_ = compile_node(node.object)
            output = allocate()
            ops.append(
                {
                    "op": "query_relation_qualifier",
                    "subject": subject,
                    "object": object_,
                    "relation": node.relation,
                    "qualifier": node.qualifier,
                    "out": output,
                }
            )
            return output
        if isinstance(node, Verify):
            source = compile_node(node.input)
            output = allocate()
            ops.append(
                {
                    "op": "verify",
                    "in": source,
                    "comparator": node.comparator,
                    "value": node.value.model_dump(mode="json"),
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
            "UNSUPPORTED_PROGRAM", f"cannot convert {type(node).__name__} to v{version}"
        )

    final_handle = compile_node(program)
    ops.append({"op": "emit", "in": final_handle})
    script = GraphScript.model_validate({"version": version, "ops": ops})
    allowed = frozenset(graphscript_operators(script.version))
    unavailable = sorted({op.op for op in script.ops if op.op not in allowed})
    if unavailable:
        raise GraphScriptError(
            "OP_NOT_IN_PROFILE",
            f"operators unavailable in GraphScript v{version}: {', '.join(unavailable)}",
        )
    return script


def execute_graphscript(
    script: GraphScript,
    backend: GraphBackend,
    *,
    seed_entity: str | None = None,
    allowed_relations: frozenset[str],
    max_edge_visits: int,
    max_returned_entities: int = 1_000,
    trace_id: str | None = None,
    capture_steps: bool = False,
    trace_preview_limit: int = 8,
) -> GraphScriptExecution:
    if not 1 <= trace_preview_limit <= 100:
        raise ValueError("trace_preview_limit must be between 1 and 100")
    handles: dict[str, _Handle] = {}
    support: set[Triple] = set()
    relation_path: list[str] = []
    execution_steps: list[ExecutionStepTrace] = []
    trace_output_entities: list[set[str]] = []
    trace_evidence: list[tuple[Triple, ...]] = []
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

    def require_catalog(*relation_ids: str) -> None:
        missing = sorted(set(relation_ids) - allowed_relations)
        if missing:
            raise GraphScriptError(
                "RELATION_NOT_ALLOWED",
                f"relation or qualifier is not in the episode catalog: {', '.join(missing)}",
            )

    for index, op in enumerate(script.ops):
        op_trace = f"{trace_id or 'graphscript'}:{index}"
        step_started = perf_counter()
        input_names = _input_handle_names(op)
        input_trace = (
            {
                name: _handle_trace(handles[name], preview_limit=trace_preview_limit)
                for name in input_names
            }
            if capture_steps
            else {}
        )
        input_entities = (
            set().union(*(_handle_entity_ids(handles[name]) for name in input_names))
            if input_names
            else set()
        )
        support_before = set(support)
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
            require_catalog(op.relation)
            source = handles[op.input_handle]
            program = FilterLiteral(
                input=_require_program(source),
                relation=op.relation,
                comparator=op.comparator,
                value=LiteralValue.model_validate(op.value.model_dump()),
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, FilterQualifierOp):
            require_catalog(op.qualifier)
            source = handles[op.input_handle]
            program = FilterQualifier(
                input=_require_program(source),
                qualifier=op.qualifier,
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
            require_catalog(op.attribute)
            source = handles[op.input_handle]
            program = QueryAttribute(input=_require_program(source), attribute=op.attribute)
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, QueryAttributeUnderConditionOp):
            require_catalog(op.attribute, op.qualifier)
            source = handles[op.input_handle]
            program = QueryAttributeUnderCondition(
                input=_require_program(source),
                attribute=op.attribute,
                qualifier=op.qualifier,
                qualifier_value=LiteralValue.model_validate(op.qualifier_value.model_dump()),
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, QueryAttributeQualifierOp):
            require_catalog(op.attribute, op.qualifier)
            source = handles[op.input_handle]
            program = QueryAttributeQualifier(
                input=_require_program(source),
                attribute=op.attribute,
                attribute_value=LiteralValue.model_validate(op.attribute_value.model_dump()),
                qualifier=op.qualifier,
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, QueryRelationOp):
            subject = handles[op.subject]
            object_ = handles[op.object]
            program = QueryRelation(
                subject=_require_program(subject), object=_require_program(object_)
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, QueryRelationQualifierOp):
            require_catalog(op.relation, op.qualifier)
            subject = handles[op.subject]
            object_ = handles[op.object]
            program = QueryRelationQualifier(
                subject=_require_program(subject),
                object=_require_program(object_),
                relation=op.relation,
                qualifier=op.qualifier,
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, VerifyOp):
            source = handles[op.input_handle]
            program = Verify(
                input=_require_program(source),
                comparator=op.comparator,
                value=LiteralValue.model_validate(op.value.model_dump()),
            )
            handles[op.out] = _Handle(program=program, answers=execute_program(program))
        elif isinstance(op, SelectBetweenOp):
            require_catalog(op.attribute)
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
            require_catalog(op.attribute)
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

        if capture_steps:
            output_handle = getattr(op, "out", None)
            if isinstance(output_handle, str):
                output_source = handles[output_handle]
            elif isinstance(op, RequireUniqueOp | EmitOp):
                output_source = handles[op.input_handle]
            else:
                output_source = None
            output_trace = (
                _handle_trace(output_source, preview_limit=trace_preview_limit)
                if output_source is not None
                else None
            )
            output_entities = (
                _handle_entity_ids(output_source) if output_source is not None else set()
            )
            retrieved = (
                output_entities
                if isinstance(
                    op,
                    StartOp | ResolveEntityOp | SearchPassageOp | PassagePagesOp | FollowOp,
                )
                else set()
            )
            discarded = (
                input_entities - output_entities
                if isinstance(
                    op,
                    FilterTypeOp
                    | FilterLiteralOp
                    | FilterQualifierOp
                    | IntersectOp
                    | SelectBetweenOp
                    | SelectAmongOp,
                )
                else set()
            )
            evidence = tuple(sorted(support - support_before, key=Triple.sort_key))
            execution_steps.append(
                ExecutionStepTrace(
                    index=index,
                    operation=op.model_dump(mode="json", by_alias=True),
                    input_handles=input_trace,
                    output_handle=output_handle if isinstance(output_handle, str) else None,
                    output=output_trace,
                    selected_entities=_bounded(output_entities, trace_preview_limit),
                    retrieved_entities=_bounded(retrieved, trace_preview_limit),
                    discarded_entities=_bounded(discarded, trace_preview_limit),
                    new_evidence=evidence[:trace_preview_limit],
                    new_evidence_total=len(evidence),
                    evidence_truncated=len(evidence) > trace_preview_limit,
                    latency_ms=(perf_counter() - step_started) * 1_000,
                    cumulative_usage=BudgetUsage(
                        edge_visits=edge_visits,
                        operators=index + 1,
                        returned_entities=returned_entities,
                        graph_calls=graph_calls,
                        passage_searches=passage_searches,
                        returned_passages=returned_passages,
                    ),
                )
            )
            trace_output_entities.append(output_entities)
            trace_evidence.append(evidence)

    if emitted is None or emitted.answers is None or emitted.program is None:
        raise GraphScriptError("MISSING_EMIT", "script did not emit an executable answer")
    if not emitted.answers.answers:
        raise GraphScriptError("EMPTY_RESULT", "emitted answer is empty")
    if capture_steps:
        downstream_relevant: set[str] = set()
        for index in reversed(range(len(execution_steps))):
            output_entities = trace_output_entities[index]
            if 0 < len(output_entities) <= trace_preview_limit:
                downstream_relevant.update(output_entities)
            op = script.ops[index]
            input_names = _input_handle_names(op)
            output_handle = getattr(op, "out", None)
            if isinstance(output_handle, str):
                output_source = handles[output_handle]
            elif isinstance(op, RequireUniqueOp | EmitOp):
                output_source = handles[op.input_handle]
            else:
                output_source = None
            evidence = trace_evidence[index]
            ordered_evidence = tuple(
                sorted(
                    evidence,
                    key=lambda triple: (
                        0
                        if triple.subject in downstream_relevant
                        or triple.object in downstream_relevant
                        else 1,
                        triple.sort_key(),
                    ),
                )
            )
            step = execution_steps[index]
            execution_steps[index] = step.model_copy(
                update={
                    "input_handles": {
                        name: _handle_trace(
                            handles[name],
                            preview_limit=trace_preview_limit,
                            preferred=downstream_relevant,
                        )
                        for name in input_names
                    },
                    "output": (
                        _handle_trace(
                            output_source,
                            preview_limit=trace_preview_limit,
                            preferred=downstream_relevant,
                        )
                        if output_source is not None
                        else None
                    ),
                    "selected_entities": _bounded(
                        output_entities,
                        trace_preview_limit,
                        preferred=downstream_relevant,
                    ),
                    "retrieved_entities": _bounded(
                        set(step.retrieved_entities) | output_entities,
                        trace_preview_limit,
                        preferred=downstream_relevant,
                    )
                    if step.retrieved_entities
                    else (),
                    "new_evidence": ordered_evidence[:trace_preview_limit],
                    "evidence_truncated": len(ordered_evidence) > trace_preview_limit,
                }
            )
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
        steps=tuple(execution_steps),
    )
