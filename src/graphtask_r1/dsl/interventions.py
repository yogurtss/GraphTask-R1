from __future__ import annotations

from dataclasses import dataclass

from graphtask_r1.graph import GraphBackend, GraphOverlay
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    Triple,
    Union,
)


@dataclass(frozen=True)
class ProgramIntervention:
    code: str
    program: Program


def atomic_interventions(program: Program) -> list[ProgramIntervention]:
    results: list[ProgramIntervention] = []
    if isinstance(program, FilterType | FilterLiteral | Hop):
        results.append(ProgramIntervention(f"bypass_{program.op}", program.input))
        for child in atomic_interventions(program.input):
            results.append(
                ProgramIntervention(child.code, program.model_copy(update={"input": child.program}))
            )
    elif isinstance(program, Count):
        for child in atomic_interventions(program.input):
            results.append(
                ProgramIntervention(child.code, program.model_copy(update={"input": child.program}))
            )
    elif isinstance(program, Intersect | Union):
        for index in range(len(program.inputs)):
            remaining = tuple(branch for i, branch in enumerate(program.inputs) if i != index)
            replacement: Program
            if len(remaining) == 1:
                replacement = remaining[0]
            elif isinstance(program, Intersect):
                replacement = Intersect(inputs=remaining)
            else:
                replacement = Union(inputs=remaining)
            family = "intersection" if isinstance(program, Intersect) else "union"
            results.append(ProgramIntervention(f"drop_{family}_branch_{index}", replacement))
        for index, branch in enumerate(program.inputs):
            for child in atomic_interventions(branch):
                branches = list(program.inputs)
                branches[index] = child.program
                results.append(
                    ProgramIntervention(
                        child.code, program.model_copy(update={"inputs": tuple(branches)})
                    )
                )
    elif not isinstance(program, Entity | AllEntities):
        raise TypeError(type(program).__name__)
    return results


def jaccard(left: AnswerSet, right: AnswerSet) -> float:
    left_values = {(answer.kind, str(answer.value)) for answer in left.answers}
    right_values = {(answer.kind, str(answer.value)) for answer in right.answers}
    if not left_values and not right_values:
        return 1.0
    return len(left_values & right_values) / len(left_values | right_values)


def necessity_scores(
    program: Program, backend: GraphBackend
) -> tuple[float, float, dict[str, float]]:
    gold = backend.execute_program(program)
    interventions = atomic_interventions(program)
    if not interventions:
        return 1.0, 1.0, {}
    scores = {
        f"{item.code}:{index}": 1.0 - jaccard(gold, backend.execute_program(item.program))
        for index, item in enumerate(interventions)
    }
    return sum(scores.values()) / len(scores), min(scores.values()), scores


def remove_witness_facts(backend: GraphBackend, facts: tuple[Triple, ...]) -> GraphBackend:
    return backend.with_overlay(GraphOverlay(removed=facts))
