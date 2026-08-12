from __future__ import annotations

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

WEIGHTS = {
    "entity": 0.0,
    "all_entities": 0.5,
    "hop": 1.0,
    "filter_type": 0.5,
    "filter_literal": 1.0,
    "filter_qualifier": 1.0,
    "intersect": 1.5,
    "union": 1.5,
    "count": 1.0,
    "query_attribute": 1.0,
    "query_attribute_under_condition": 1.5,
    "query_attribute_qualifier": 1.5,
    "query_relation": 1.0,
    "query_relation_qualifier": 1.5,
    "verify": 0.5,
    "select_among": 1.5,
    "select_between": 1.5,
}


def program_cost(program: Program) -> float:
    if isinstance(program, Entity | AllEntities):
        return WEIGHTS[program.op]
    if isinstance(program, Hop):
        return WEIGHTS[program.op] + program_cost(program.input)
    if isinstance(program, Intersect | Union):
        return WEIGHTS[program.op] + sum(program_cost(branch) for branch in program.inputs)
    if isinstance(
        program,
        FilterType
        | FilterLiteral
        | FilterQualifier
        | Count
        | QueryAttribute
        | QueryAttributeUnderCondition
        | QueryAttributeQualifier
        | Verify
        | SelectAmong,
    ):
        return WEIGHTS[program.op] + program_cost(program.input)
    if isinstance(program, QueryRelation | QueryRelationQualifier):
        return WEIGHTS[program.op] + program_cost(program.subject) + program_cost(program.object)
    if isinstance(program, SelectBetween):
        return WEIGHTS[program.op] + program_cost(program.left) + program_cost(program.right)
    raise TypeError(type(program).__name__)
