from __future__ import annotations

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
    Union,
)


def _phrase(program: Program, backend: GraphBackend) -> str:
    if isinstance(program, Entity):
        return backend.entity_info(program.entity_id).label
    if isinstance(program, AllEntities):
        return "entities in the graph"
    if isinstance(program, Hop):
        relation = backend.relation_info(program.relation).label
        direction = "from" if program.direction == "out" else "to"
        return f"entities connected by {relation} {direction} {_phrase(program.input, backend)}"
    if isinstance(program, Intersect):
        return " and ".join(_phrase(branch, backend) for branch in program.inputs)
    if isinstance(program, Union):
        return " or ".join(_phrase(branch, backend) for branch in program.inputs)
    if isinstance(program, FilterType):
        return f"{_phrase(program.input, backend)} that are of type {program.type_id}"
    if isinstance(program, FilterLiteral):
        return (
            f"{_phrase(program.input, backend)} whose {program.relation} is "
            f"{program.comparator} {program.value.value}"
        )
    if isinstance(program, Count):
        return _phrase(program.input, backend)
    raise TypeError(type(program).__name__)


def verbalize(program: Program, backend: GraphBackend) -> str:
    phrase = _phrase(program, backend)
    if isinstance(program, Count):
        return f"How many {phrase} are there?"
    return f"Which {phrase}?"
