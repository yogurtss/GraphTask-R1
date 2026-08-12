from __future__ import annotations

from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    AllEntities,
    Count,
    Entity,
    FilterLiteral,
    FilterQualifier,
    FilterType,
    Hop,
    Intersect,
    Program,
    QueryAttribute,
    QueryAttributeQualifier,
    QueryAttributeUnderCondition,
    QueryRelation,
    QueryRelationQualifier,
    SelectAmong,
    SelectBetween,
    Union,
    Verify,
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
    if isinstance(program, FilterQualifier):
        return (
            f"{_phrase(program.input, backend)} whose fact qualifier {program.qualifier} is "
            f"{program.comparator} {program.value.value}"
        )
    if isinstance(program, Count):
        return _phrase(program.input, backend)
    if isinstance(program, QueryAttribute):
        return f"the {program.attribute} of {_phrase(program.input, backend)}"
    if isinstance(program, QueryAttributeUnderCondition):
        return (
            f"the {program.attribute} of {_phrase(program.input, backend)} where "
            f"{program.qualifier} is {program.qualifier_value.value}"
        )
    if isinstance(program, QueryAttributeQualifier):
        return (
            f"the {program.qualifier} qualifier of {program.attribute} "
            f"{program.attribute_value.value} for {_phrase(program.input, backend)}"
        )
    if isinstance(program, QueryRelation):
        return (
            f"the relation from {_phrase(program.subject, backend)} "
            f"to {_phrase(program.object, backend)}"
        )
    if isinstance(program, QueryRelationQualifier):
        return (
            f"the {program.qualifier} qualifier of {program.relation} from "
            f"{_phrase(program.subject, backend)} to {_phrase(program.object, backend)}"
        )
    if isinstance(program, Verify):
        return (
            f"whether {_phrase(program.input, backend)} is {program.comparator} "
            f"{program.value.value}"
        )
    if isinstance(program, SelectAmong):
        extreme = "smallest" if program.mode == "min" else "largest"
        return f"the {extreme} by {program.attribute} among {_phrase(program.input, backend)}"
    if isinstance(program, SelectBetween):
        extreme = "smaller" if program.mode == "min" else "larger"
        return (
            f"the {extreme} by {program.attribute} between "
            f"{_phrase(program.left, backend)} and {_phrase(program.right, backend)}"
        )
    raise TypeError(type(program).__name__)


def verbalize(program: Program, backend: GraphBackend) -> str:
    phrase = _phrase(program, backend)
    if isinstance(program, Count):
        return f"How many {phrase} are there?"
    if isinstance(
        program,
        QueryAttribute
        | QueryAttributeUnderCondition
        | QueryAttributeQualifier
        | QueryRelation
        | QueryRelationQualifier,
    ):
        return f"What is {phrase}?"
    if isinstance(program, Verify):
        return f"Is it true that {phrase}?"
    return f"Which {phrase}?"
