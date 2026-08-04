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
    raise TypeError(type(program).__name__)


def canonicalize(program: Program) -> Program:
    if isinstance(program, Entity | AllEntities):
        return program
    if isinstance(program, Hop):
        return program.model_copy(update={"input": canonicalize(program.input)})
    if isinstance(program, Intersect | Union):
        branches = tuple(sorted((canonicalize(p) for p in program.inputs), key=canonical_signature))
        return program.model_copy(update={"inputs": branches})
    if isinstance(program, FilterType | FilterLiteral | Count):
        return program.model_copy(update={"input": canonicalize(program.input)})
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
        elif isinstance(node, FilterType | FilterLiteral | Count):
            visit(node.input)

    visit(program)
    return tuple(sorted(tags))
