from __future__ import annotations

import base64
import json
import re

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
    program_to_dict,
)


def escape_iri(value: str) -> str:
    if any(char in value for char in '<>"{}|\\^`\n\r'):
        raise ValueError(f"unsafe IRI token: {value!r}")
    return value


def _sparql_literal(program: FilterLiteral) -> str:
    value = program.value.value
    if program.value.datatype in {"quantity", "number"} and isinstance(value, int | float):
        return str(value)
    if program.value.datatype == "year":
        return str(int(value))
    if program.value.datatype == "date":
        normalized = str(value).replace("/", "-")
        if not re.fullmatch(r"-?\d{1,6}-\d{1,2}-\d{1,2}", normalized):
            raise ValueError(f"invalid date literal: {value!r}")
        return json.dumps(normalized) + "^^<http://www.w3.org/2001/XMLSchema#date>"
    return json.dumps(value, ensure_ascii=False)


class _Builder:
    def __init__(self) -> None:
        self.index = 0

    def var(self) -> str:
        value = f"?v{self.index}"
        self.index += 1
        return value

    def compile(self, program: Program) -> tuple[str, list[str]]:
        if isinstance(program, AllEntities):
            variable = self.var()
            return variable, [f"{variable} ?allRelation ?allObject ."]
        if isinstance(program, Entity):
            variable = self.var()
            return variable, [f"VALUES {variable} {{ <{escape_iri(program.entity_id)}> }}"]
        if isinstance(program, Hop):
            source, input_clauses = self.compile(program.input)
            target = self.var()
            relation = escape_iri(program.relation)
            triple = (
                f"{source} <{relation}> {target} ."
                if program.direction == "out"
                else f"{target} <{relation}> {source} ."
            )
            return target, [*input_clauses, triple]
        if isinstance(program, Intersect):
            compiled = [self.compile(branch) for branch in program.inputs]
            output = self.var()
            intersection_clauses: list[str] = []
            for variable, branch_clauses in compiled:
                intersection_clauses.extend(branch_clauses)
                intersection_clauses.append(f"BIND({variable} AS {output})")
            return output, intersection_clauses
        if isinstance(program, Union):
            compiled = [self.compile(branch) for branch in program.inputs]
            output = self.var()
            branches = []
            for variable, branch_clauses in compiled:
                body = " ".join([*branch_clauses, f"BIND({variable} AS {output})"])
                branches.append(f"{{ {body} }}")
            return output, [" UNION ".join(branches)]
        if isinstance(program, FilterType):
            variable, input_clauses = self.compile(program.input)
            return variable, [
                *input_clauses,
                f"{variable} a <{escape_iri(program.type_id)}> .",
            ]
        if isinstance(program, FilterLiteral):
            variable, input_clauses = self.compile(program.input)
            literal = self.var()
            value = _sparql_literal(program)
            operators = {"eq": "=", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
            if program.comparator == "contains":
                expression = f"CONTAINS(LCASE(STR({literal})), LCASE(STR({value})))"
            else:
                expression = f"{literal} {operators[program.comparator]} {value}"
            return variable, [
                *input_clauses,
                f"{variable} <{escape_iri(program.relation)}> {literal} .",
                f"FILTER({expression})",
            ]
        if isinstance(program, Count):
            return self.compile(program.input)
        if isinstance(program, QueryAttribute):
            entity, input_clauses = self.compile(program.input)
            value = self.var()
            return value, [
                *input_clauses,
                f"{entity} <{escape_iri(program.attribute)}> {value} .",
            ]
        if isinstance(program, QueryRelation):
            subject, subject_clauses = self.compile(program.subject)
            object_, object_clauses = self.compile(program.object)
            relation = self.var()
            return relation, [
                *subject_clauses,
                *object_clauses,
                f"{subject} {relation} {object_} .",
            ]
        if isinstance(program, SelectAmong):
            entity, input_clauses = self.compile(program.input)
            value = self.var()
            direction = "ASC" if program.mode == "min" else "DESC"
            body = " ".join(
                [
                    *input_clauses,
                    f"{entity} <{escape_iri(program.attribute)}> {value} .",
                ]
            )
            return entity, [
                f"{{ SELECT DISTINCT {entity} WHERE {{ {body} }} "
                f"ORDER BY {direction}({value}) {entity} LIMIT 1 }}"
            ]
        if isinstance(program, SelectBetween):
            left, left_clauses = self.compile(program.left)
            right, right_clauses = self.compile(program.right)
            entity = self.var()
            value = self.var()
            union_body = " UNION ".join(
                [
                    "{ " + " ".join([*left_clauses, f"BIND({left} AS {entity})"]) + " }",
                    "{ " + " ".join([*right_clauses, f"BIND({right} AS {entity})"]) + " }",
                ]
            )
            direction = "ASC" if program.mode == "min" else "DESC"
            body = f"{union_body} {entity} <{escape_iri(program.attribute)}> {value} ."
            return entity, [
                f"{{ SELECT DISTINCT {entity} WHERE {{ {body} }} "
                f"ORDER BY {direction}({value}) {entity} LIMIT 1 }}"
            ]
        raise TypeError(type(program).__name__)


def compile_sparql(program: Program) -> str:
    raw = json.dumps(program_to_dict(program), sort_keys=True, separators=(",", ":"))
    marker = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    builder = _Builder()
    variable, clauses = builder.compile(program)
    body = "\n  ".join(clauses)
    if isinstance(program, Count):
        select = f"SELECT (COUNT(DISTINCT {variable}) AS ?count)"
    else:
        select = f"SELECT DISTINCT {variable}"
    return f"# graphtask-program:{marker}\n{select} WHERE {{\n  {body}\n}} ORDER BY {variable}"
