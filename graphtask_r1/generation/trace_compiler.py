from __future__ import annotations

import hashlib

from pydantic import ValidationError

from graphtask_r1.envs import SolverEnv
from graphtask_r1.envs.graph_query import (
    MAX_COMPACT_QUERY_ENTITIES,
    MAX_COMPACT_QUERY_STEPS,
    AllEntitiesQueryRoot,
    CompactGraphQuery,
    EntityQueryRoot,
    FilterLiteralQueryStep,
    FilterTypeQueryStep,
    HopQueryStep,
    QueryAttributeQueryStep,
    QueryRelationQueryStep,
    QueryStep,
    SelectAttributeQueryStep,
    execute_compact_query,
)
from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Count,
    Entity,
    EpisodeInput,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    QueryAttribute,
    QueryRelation,
    SelectAmong,
    SelectBetween,
    ToolCall,
    Trajectory,
    Union,
)


class TraceCompilationError(RuntimeError):
    """A structured rejection raised before an oversized trace is serialized."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code


def _trace_id(task_id: str, index: int) -> str:
    return hashlib.sha256(f"{task_id}:{index}".encode()).hexdigest()[:16]


def _entity_union_root(program: Union) -> EntityQueryRoot | None:
    if not all(isinstance(branch, Entity) for branch in program.inputs):
        return None
    entity_ids = tuple(branch.entity_id for branch in program.inputs if isinstance(branch, Entity))
    if len(entity_ids) > MAX_COMPACT_QUERY_ENTITIES:
        return None
    return EntityQueryRoot(entity_ids=entity_ids)


def _type_union(program: Union) -> tuple[Program, tuple[str, ...]] | None:
    branches = program.inputs
    if not all(isinstance(branch, FilterType) for branch in branches):
        return None
    filters = tuple(branch for branch in branches if isinstance(branch, FilterType))
    first_input = filters[0].input
    if any(branch.input != first_input for branch in filters[1:]):
        return None
    return first_input, tuple(branch.type_id for branch in filters)


def _as_compact_query(program: Program) -> CompactGraphQuery | None:
    return_count = isinstance(program, Count)
    node = program.input if isinstance(program, Count) else program
    steps: list[QueryStep] = []

    while isinstance(node, Hop | FilterType | FilterLiteral | QueryAttribute | SelectAmong):
        if isinstance(node, Hop):
            steps.append(HopQueryStep(relation=node.relation, direction=node.direction))
        elif isinstance(node, FilterType):
            steps.append(FilterTypeQueryStep(type_ids=(node.type_id,)))
        elif isinstance(node, FilterLiteral):
            steps.append(
                FilterLiteralQueryStep(
                    relation=node.relation,
                    comparator=node.comparator,
                    value=node.value,
                )
            )
        elif isinstance(node, QueryAttribute):
            steps.append(QueryAttributeQueryStep(attribute=node.attribute))
        else:
            steps.append(SelectAttributeQueryStep(attribute=node.attribute, mode=node.mode))
        node = node.input

    if isinstance(node, Union):
        type_union = _type_union(node)
        if type_union is not None:
            node, type_ids = type_union
            steps.append(FilterTypeQueryStep(type_ids=type_ids))

    if isinstance(node, Entity):
        root: EntityQueryRoot | AllEntitiesQueryRoot = EntityQueryRoot(entity_ids=(node.entity_id,))
    elif isinstance(node, AllEntities):
        root = AllEntitiesQueryRoot(max_results=node.max_results)
    elif isinstance(node, Union):
        entity_root = _entity_union_root(node)
        if entity_root is None:
            return None
        root = entity_root
    else:
        return None

    steps.reverse()
    if len(steps) > MAX_COMPACT_QUERY_STEPS:
        return None
    try:
        return CompactGraphQuery(
            root=root,
            steps=tuple(steps),
            return_count=return_count,
            limit=MAX_COMPACT_QUERY_ENTITIES,
        )
    except ValidationError:
        return None


def _append_query(
    query: CompactGraphQuery,
    expected: AnswerSet,
    backend: GraphBackend,
    task_id: str,
    calls: list[ToolCall],
    *,
    max_tool_calls: int,
    max_query_results: int,
) -> None:
    if len(calls) >= max_tool_calls - 1:
        raise TraceCompilationError(
            "TRACE_TOOL_BUDGET_EXCEEDED",
            f"trace requires more than {max_tool_calls - 1} graph calls",
        )
    query = query.model_copy(update={"limit": max_query_results})
    result = execute_compact_query(backend, query, max_limit=max_query_results)
    if result.truncated:
        raise TraceCompilationError(
            "TRACE_ENTITY_BUDGET_EXCEEDED",
            f"compact query returned {result.total_entities} entities; "
            f"limit is {max_query_results}",
        )
    if query.return_count:
        observed = AnswerSet.count(result.count or 0)
    elif result.answer_kind == "literal":
        observed = AnswerSet.literals(result.values)
    else:
        observed = AnswerSet.entities(tuple(entity.entity_id for entity in result.entities))
    if observed != expected:
        raise TraceCompilationError(
            "TRACE_QUERY_MISMATCH",
            "compact graph query did not reproduce the certified program answer",
        )
    calls.append(
        ToolCall(
            name="search",
            arguments={"query": query.model_dump(mode="json")},
            trace_id=_trace_id(task_id, len(calls)),
        )
    )


def _query_from_entities(
    entity_ids: tuple[str, ...],
    step: QueryStep | None,
    *,
    return_count: bool = False,
    limit: int,
) -> CompactGraphQuery:
    if not entity_ids:
        raise TraceCompilationError(
            "TRACE_EMPTY_INTERMEDIATE",
            "cannot construct a compact query from an empty intermediate set",
        )
    if len(entity_ids) > MAX_COMPACT_QUERY_ENTITIES:
        raise TraceCompilationError(
            "TRACE_ENTITY_BUDGET_EXCEEDED",
            f"intermediate set contains {len(entity_ids)} entities; "
            f"limit is {MAX_COMPACT_QUERY_ENTITIES}",
        )
    return CompactGraphQuery(
        root=EntityQueryRoot(entity_ids=entity_ids),
        steps=() if step is None else (step,),
        return_count=return_count,
        limit=limit,
    )


def _compile_searches(
    program: Program,
    backend: GraphBackend,
    task_id: str,
    calls: list[ToolCall],
    *,
    max_tool_calls: int,
    max_query_results: int,
) -> AnswerSet:
    expected = backend.execute_program(program)
    compact = _as_compact_query(program)
    if compact is not None and not isinstance(program, Entity):
        _append_query(
            compact,
            expected,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        )
        return expected

    if isinstance(program, Entity):
        return expected
    if isinstance(program, AllEntities):
        raise TraceCompilationError(
            "TRACE_UNBOUNDED_ALL_ENTITIES",
            "AllEntities must be followed by a server-side filter before trace compilation",
        )
    if isinstance(program, Intersect | Union):
        for branch in program.inputs:
            branch_answers = _compile_searches(
                branch,
                backend,
                task_id,
                calls,
                max_tool_calls=max_tool_calls,
                max_query_results=max_query_results,
            )
            if len(branch_answers.entity_ids()) > max_query_results:
                raise TraceCompilationError(
                    "TRACE_ENTITY_BUDGET_EXCEEDED",
                    f"set branch returned {len(branch_answers.entity_ids())} entities; "
                    f"limit is {max_query_results}",
                )
        return expected

    if isinstance(program, QueryRelation):
        subjects = _compile_searches(
            program.subject,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        ).entity_ids()
        objects = _compile_searches(
            program.object,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        ).entity_ids()
        query = _query_from_entities(
            subjects,
            QueryRelationQueryStep(object_entity_ids=objects),
            limit=max_query_results,
        )
        _append_query(
            query,
            expected,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        )
        return expected

    if isinstance(program, SelectBetween):
        left = _compile_searches(
            program.left,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        ).entity_ids()
        right = _compile_searches(
            program.right,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        ).entity_ids()
        candidates = tuple(sorted({*left, *right}))
        query = _query_from_entities(
            candidates,
            SelectAttributeQueryStep(attribute=program.attribute, mode=program.mode),
            limit=max_query_results,
        )
        _append_query(
            query,
            expected,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        )
        return expected

    if isinstance(program, Hop | FilterType | FilterLiteral | Count | QueryAttribute | SelectAmong):
        inputs = _compile_searches(
            program.input,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        )
        entity_ids = inputs.entity_ids()
        if not entity_ids:
            return expected
        if isinstance(program, Hop):
            step: QueryStep | None = HopQueryStep(
                relation=program.relation, direction=program.direction
            )
        elif isinstance(program, FilterType):
            step = FilterTypeQueryStep(type_ids=(program.type_id,))
        elif isinstance(program, FilterLiteral):
            step = FilterLiteralQueryStep(
                relation=program.relation,
                comparator=program.comparator,
                value=program.value,
            )
        elif isinstance(program, QueryAttribute):
            step = QueryAttributeQueryStep(attribute=program.attribute)
        elif isinstance(program, SelectAmong):
            step = SelectAttributeQueryStep(
                attribute=program.attribute,
                mode=program.mode,
            )
        else:
            step = None
        query = _query_from_entities(
            entity_ids,
            step,
            return_count=isinstance(program, Count),
            limit=max_query_results,
        )
        _append_query(
            query,
            expected,
            backend,
            task_id,
            calls,
            max_tool_calls=max_tool_calls,
            max_query_results=max_query_results,
        )
        return expected
    raise TypeError(type(program).__name__)


def compile_trace(
    task_id: str,
    question: str,
    program: Program,
    backend: GraphBackend,
    *,
    seed: int,
    max_tool_calls: int = 8,
    max_query_results: int = MAX_COMPACT_QUERY_ENTITIES,
) -> Trajectory:
    if max_tool_calls < 2:
        raise ValueError("max_tool_calls must allow one graph call and one final answer")
    if not 1 <= max_query_results <= MAX_COMPACT_QUERY_ENTITIES:
        raise ValueError(f"max_query_results must be between 1 and {MAX_COMPACT_QUERY_ENTITIES}")
    calls: list[ToolCall] = []
    answers = _compile_searches(
        program,
        backend,
        task_id,
        calls,
        max_tool_calls=max_tool_calls,
        max_query_results=max_query_results,
    )
    if len(answers.entity_ids()) > max_query_results:
        raise TraceCompilationError(
            "TRACE_ANSWER_BUDGET_EXCEEDED",
            f"final answer contains {len(answers.entity_ids())} entities; "
            f"limit is {max_query_results}",
        )
    calls.append(
        ToolCall(
            name="final_answer",
            arguments={"answers": list(answers.values())},
            trace_id=_trace_id(task_id, len(calls)),
        )
    )
    topic_ids = _topic_entities(program)
    env = SolverEnv(
        backend,
        max_turns=max_tool_calls,
        max_observation_entities=max_query_results,
    )
    episode = EpisodeInput(
        task_id=task_id,
        question=question,
        topic_entity_ids=topic_ids,
        gold_answers=answers,
    )
    observations = [env.reset(episode, seed)]
    for call in calls:
        observation = env.step(call).observation
        if observation.truncated:
            raise TraceCompilationError(
                "TRACE_REPLAY_TRUNCATED",
                "compact query was truncated while replaying the compiled trace",
            )
        observations.append(observation)
    if env.final_answers != answers:
        raise TraceCompilationError(
            "TRACE_REPLAY_MISMATCH",
            "compiled trace did not submit the certified program answer",
        )
    return Trajectory(
        task_id=task_id,
        role="solver",
        seed=seed,
        calls=tuple(calls),
        observations=tuple(observations),
        final_answers=env.final_answers,
    )


def _topic_entities(program: Program) -> tuple[str, ...]:
    if isinstance(program, Entity):
        return (program.entity_id,)
    if isinstance(program, AllEntities):
        return ()
    if isinstance(program, Intersect | Union):
        return tuple(
            sorted({entity for branch in program.inputs for entity in _topic_entities(branch)})
        )
    if isinstance(program, Hop | FilterType | FilterLiteral | Count | QueryAttribute | SelectAmong):
        return _topic_entities(program.input)
    if isinstance(program, QueryRelation):
        return tuple(sorted({*_topic_entities(program.subject), *_topic_entities(program.object)}))
    if isinstance(program, SelectBetween):
        return tuple(sorted({*_topic_entities(program.left), *_topic_entities(program.right)}))
    raise TypeError(type(program).__name__)
