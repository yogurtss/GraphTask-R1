from __future__ import annotations

import hashlib

from graphtask_r1.envs import SolverEnv
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
    ToolCall,
    Trajectory,
    Union,
)


def _trace_id(task_id: str, index: int) -> str:
    return hashlib.sha256(f"{task_id}:{index}".encode()).hexdigest()[:16]


def _compile_searches(
    program: Program, backend: GraphBackend, task_id: str, calls: list[ToolCall]
) -> AnswerSet:
    if isinstance(program, Entity):
        return AnswerSet.entities([program.entity_id])
    if isinstance(program, AllEntities):
        return AnswerSet.entities(backend.all_entities(limit=program.max_results))
    if isinstance(program, Hop):
        inputs = _compile_searches(program.input, backend, task_id, calls)
        calls.append(
            ToolCall(
                name="search",
                arguments={
                    "entity_ids": list(inputs.entity_ids()),
                    "direction": program.direction,
                    "relation_ids": [program.relation],
                },
                trace_id=_trace_id(task_id, len(calls)),
            )
        )
        return backend.execute_program(program)
    if isinstance(program, Intersect):
        for branch in program.inputs:
            _compile_searches(branch, backend, task_id, calls)
        return backend.execute_program(program)
    if isinstance(program, Union):
        for branch in program.inputs:
            _compile_searches(branch, backend, task_id, calls)
        return backend.execute_program(program)
    if isinstance(program, FilterType):
        inputs = _compile_searches(program.input, backend, task_id, calls)
        for entity_id in inputs.entity_ids():
            calls.append(
                ToolCall(
                    name="inspect_entity",
                    arguments={"entity_id": entity_id},
                    trace_id=_trace_id(task_id, len(calls)),
                )
            )
        return backend.execute_program(program)
    if isinstance(program, FilterLiteral):
        inputs = _compile_searches(program.input, backend, task_id, calls)
        calls.append(
            ToolCall(
                name="search",
                arguments={
                    "entity_ids": list(inputs.entity_ids()),
                    "direction": "out",
                    "relation_ids": [program.relation],
                },
                trace_id=_trace_id(task_id, len(calls)),
            )
        )
        return backend.execute_program(program)
    if isinstance(program, Count):
        _compile_searches(program.input, backend, task_id, calls)
        return backend.execute_program(program)
    raise TypeError(type(program).__name__)


def compile_trace(
    task_id: str,
    question: str,
    program: Program,
    backend: GraphBackend,
    *,
    seed: int,
) -> Trajectory:
    calls: list[ToolCall] = []
    answers = _compile_searches(program, backend, task_id, calls)
    calls.append(
        ToolCall(
            name="final_answer",
            arguments={"answers": list(answers.values())},
            trace_id=_trace_id(task_id, len(calls)),
        )
    )
    topic_ids = _topic_entities(program)
    env = SolverEnv(backend, max_turns=max(8, len(calls) + 1))
    episode = EpisodeInput(
        task_id=task_id,
        question=question,
        topic_entity_ids=topic_ids,
        gold_answers=answers,
    )
    observations = [env.reset(episode, seed)]
    for call in calls:
        observations.append(env.step(call).observation)
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
    if isinstance(program, Hop | FilterType | FilterLiteral | Count):
        return _topic_entities(program.input)
    raise TypeError(type(program).__name__)
