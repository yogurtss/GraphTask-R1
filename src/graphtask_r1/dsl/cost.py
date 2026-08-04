from __future__ import annotations

from graphtask_r1.schema import Count, Entity, FilterLiteral, FilterType, Hop, Intersect, Program

WEIGHTS = {
    "entity": 0.0,
    "hop": 1.0,
    "filter_type": 0.5,
    "filter_literal": 1.0,
    "intersect": 1.5,
    "count": 1.0,
}


def program_cost(program: Program) -> float:
    if isinstance(program, Entity):
        return 0.0
    if isinstance(program, Hop):
        return WEIGHTS[program.op] + program_cost(program.input)
    if isinstance(program, Intersect):
        return WEIGHTS[program.op] + sum(program_cost(branch) for branch in program.inputs)
    if isinstance(program, FilterType | FilterLiteral | Count):
        return WEIGHTS[program.op] + program_cost(program.input)
    raise TypeError(type(program).__name__)
