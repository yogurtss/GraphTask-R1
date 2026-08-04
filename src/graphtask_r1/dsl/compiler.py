from __future__ import annotations

import base64
import json

from graphtask_r1.schema import (
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    program_to_dict,
)


def escape_iri(value: str) -> str:
    if any(char in value for char in '<>"{}|\\^`\n\r'):
        raise ValueError(f"unsafe IRI token: {value!r}")
    return value


class _Builder:
    def __init__(self) -> None:
        self.index = 0

    def var(self) -> str:
        value = f"?v{self.index}"
        self.index += 1
        return value

    def compile(self, program: Program) -> tuple[str, list[str]]:
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
        if isinstance(program, FilterType):
            variable, input_clauses = self.compile(program.input)
            return variable, [
                *input_clauses,
                f"{variable} a <{escape_iri(program.type_id)}> .",
            ]
        if isinstance(program, FilterLiteral):
            variable, input_clauses = self.compile(program.input)
            literal = self.var()
            value = json.dumps(program.value, ensure_ascii=False)
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
