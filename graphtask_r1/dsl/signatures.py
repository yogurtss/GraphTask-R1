from __future__ import annotations

from graphtask_r1.schema import (
    AllEntities,
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    QueryAttribute,
    QueryRelation,
    SelectAmong,
    SelectBetween,
    Union,
)


def canonical_signature(program: Program) -> str:
    if isinstance(program, Entity):
        return f"entity({program.entity_id})"
    if isinstance(program, AllEntities):
        return f"all_entities({program.max_results})"
    if isinstance(program, Hop):
        return f"hop({canonical_signature(program.input)},{program.relation},{program.direction})"
    if isinstance(program, Intersect):
        branches = sorted(canonical_signature(branch) for branch in program.inputs)
        return f"intersect({','.join(branches)})"
    if isinstance(program, Union):
        branches = sorted(canonical_signature(branch) for branch in program.inputs)
        return f"union({','.join(branches)})"
    if isinstance(program, FilterType):
        return f"filter_type({canonical_signature(program.input)},{program.type_id})"
    if isinstance(program, FilterLiteral):
        return (
            f"filter_literal({canonical_signature(program.input)},{program.relation},"
            f"{program.comparator},{program.value.model_dump_json()})"
        )
    if isinstance(program, Count):
        return f"count({canonical_signature(program.input)})"
    if isinstance(program, QueryAttribute):
        return f"query_attribute({canonical_signature(program.input)},{program.attribute})"
    if isinstance(program, QueryRelation):
        return (
            f"query_relation({canonical_signature(program.subject)},"
            f"{canonical_signature(program.object)})"
        )
    if isinstance(program, SelectAmong):
        return (
            f"select_among({canonical_signature(program.input)},{program.attribute},{program.mode})"
        )
    if isinstance(program, SelectBetween):
        return (
            f"select_between({canonical_signature(program.left)},"
            f"{canonical_signature(program.right)},{program.attribute},{program.mode})"
        )
    raise TypeError(type(program).__name__)


def canonicalize(program: Program) -> Program:
    if isinstance(program, Entity | AllEntities):
        return program
    if isinstance(program, Hop):
        return program.model_copy(update={"input": canonicalize(program.input)})
    if isinstance(program, Intersect | Union):
        branches = tuple(sorted((canonicalize(p) for p in program.inputs), key=canonical_signature))
        return program.model_copy(update={"inputs": branches})
    if isinstance(program, FilterType | FilterLiteral | Count | QueryAttribute | SelectAmong):
        return program.model_copy(update={"input": canonicalize(program.input)})
    if isinstance(program, QueryRelation):
        return program.model_copy(
            update={
                "subject": canonicalize(program.subject),
                "object": canonicalize(program.object),
            }
        )
    if isinstance(program, SelectBetween):
        return program.model_copy(
            update={"left": canonicalize(program.left), "right": canonicalize(program.right)}
        )
    raise TypeError(type(program).__name__)


def operator_tags(program: Program) -> tuple[str, ...]:
    tags: set[str] = set()

    def visit(node: Program) -> None:
        tags.add(node.op)
        if isinstance(node, Hop):
            visit(node.input)
        elif isinstance(node, Intersect | Union):
            for branch in node.inputs:
                visit(branch)
        elif isinstance(node, FilterType | FilterLiteral | Count | QueryAttribute | SelectAmong):
            visit(node.input)
        elif isinstance(node, QueryRelation):
            visit(node.subject)
            visit(node.object)
        elif isinstance(node, SelectBetween):
            visit(node.left)
            visit(node.right)

    visit(program)
    return tuple(sorted(tags))
