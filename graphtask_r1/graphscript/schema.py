from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

HANDLE_PATTERN = re.compile(r"^h(?:[0-9]|[1-5][0-9]|6[0-3])$")
GRAPHSCRIPT_V01_OPERATORS = ("start", "follow", "require_unique", "emit")
GRAPHSCRIPT_V02_OPERATORS = (
    "start",
    "all_entities",
    "resolve_entity",
    "search_passage",
    "passage_pages",
    "follow",
    "intersect",
    "union",
    "filter_type",
    "filter_literal",
    "count",
    "query_attribute",
    "query_relation",
    "select_between",
    "select_among",
    "require_unique",
    "emit",
)


def graphscript_operators(version: Literal["0.1", "0.2"]) -> tuple[str, ...]:
    return GRAPHSCRIPT_V01_OPERATORS if version == "0.1" else GRAPHSCRIPT_V02_OPERATORS


class GraphScriptError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class StartOp(_Op):
    op: Literal["start"] = "start"
    entity: Literal["$seed"] = "$seed"
    out: str


class AllEntitiesOp(_Op):
    op: Literal["all_entities"] = "all_entities"
    max_results: int = Field(default=1_000_000, gt=0, le=1_000_000)
    out: str


class ResolveEntityOp(_Op):
    op: Literal["resolve_entity"] = "resolve_entity"
    query: str = Field(min_length=1, max_length=256)
    match: Literal["id", "exact", "search"] = "exact"
    limit: int = Field(default=5, gt=0, le=20)
    out: str


class SearchPassageOp(_Op):
    op: Literal["search_passage"] = "search_passage"
    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=3, gt=0, le=10)
    max_chars: int = Field(default=2_000, gt=0, le=4_000)
    out: str


class PassagePagesOp(_Op):
    op: Literal["passage_pages"] = "passage_pages"
    input_handle: str = Field(alias="in")
    out: str


class FollowOp(_Op):
    op: Literal["follow"] = "follow"
    input_handle: str = Field(alias="in")
    relation: str = Field(min_length=1)
    direction: Literal["out", "in"] = "out"
    limit: int = Field(gt=0)
    out: str


class IntersectOp(_Op):
    op: Literal["intersect"] = "intersect"
    inputs: tuple[str, ...] = Field(min_length=2)
    out: str


class UnionOp(_Op):
    op: Literal["union"] = "union"
    inputs: tuple[str, ...] = Field(min_length=2)
    out: str


class FilterTypeOp(_Op):
    op: Literal["filter_type"] = "filter_type"
    input_handle: str = Field(alias="in")
    type_id: str = Field(min_length=1)
    out: str


class ScriptLiteralValue(_Op):
    value: str | int | float
    datatype: Literal["string", "quantity", "year", "date", "number"] = "string"
    unit: str | None = None


class FilterLiteralOp(_Op):
    op: Literal["filter_literal"] = "filter_literal"
    input_handle: str = Field(alias="in")
    relation: str = Field(min_length=1)
    comparator: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains"] = "eq"
    value: ScriptLiteralValue
    out: str


class CountOp(_Op):
    op: Literal["count"] = "count"
    input_handle: str = Field(alias="in")
    out: str


class QueryAttributeOp(_Op):
    op: Literal["query_attribute"] = "query_attribute"
    input_handle: str = Field(alias="in")
    attribute: str = Field(min_length=1)
    out: str


class QueryRelationOp(_Op):
    op: Literal["query_relation"] = "query_relation"
    subject: str
    object: str
    out: str


class SelectBetweenOp(_Op):
    op: Literal["select_between"] = "select_between"
    left: str
    right: str
    attribute: str = Field(min_length=1)
    mode: Literal["min", "max"]
    out: str


class SelectAmongOp(_Op):
    op: Literal["select_among"] = "select_among"
    input_handle: str = Field(alias="in")
    attribute: str = Field(min_length=1)
    mode: Literal["min", "max"]
    out: str


class RequireUniqueOp(_Op):
    op: Literal["require_unique"] = "require_unique"
    input_handle: str = Field(alias="in")


class EmitOp(_Op):
    op: Literal["emit"] = "emit"
    input_handle: str = Field(alias="in")


GraphScriptOp: TypeAlias = Annotated[
    StartOp
    | AllEntitiesOp
    | ResolveEntityOp
    | SearchPassageOp
    | PassagePagesOp
    | FollowOp
    | IntersectOp
    | UnionOp
    | FilterTypeOp
    | FilterLiteralOp
    | CountOp
    | QueryAttributeOp
    | QueryRelationOp
    | SelectBetweenOp
    | SelectAmongOp
    | RequireUniqueOp
    | EmitOp,
    Field(discriminator="op"),
]
OP_ADAPTER: TypeAdapter[GraphScriptOp] = TypeAdapter(GraphScriptOp)


class GraphScript(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["0.1", "0.2"] = "0.1"
    ops: tuple[GraphScriptOp, ...] = Field(min_length=1, max_length=64)


class BudgetUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_visits: int = Field(default=0, ge=0)
    operators: int = Field(default=0, ge=0)
    returned_entities: int = Field(default=0, ge=0)
    graph_calls: int = Field(default=0, ge=0)
    passage_searches: int = Field(default=0, ge=0)
    returned_passages: int = Field(default=0, ge=0)


def _validation_reason(exc: ValidationError) -> str:
    errors = exc.errors()
    if any(error["type"] == "extra_forbidden" for error in errors):
        return "EXTRA_FIELD"
    if any("version" in error["loc"] for error in errors):
        return "UNSUPPORTED_VERSION"
    if any(error["type"] == "union_tag_invalid" for error in errors):
        return "UNKNOWN_OP"
    if any("direction" in error["loc"] for error in errors):
        return "INVALID_DIRECTION"
    if any("limit" in error["loc"] for error in errors):
        return "LIMIT_EXCEEDED"
    return "INVALID_SCHEMA"


def _decode_exact_json(text: str) -> Any:
    stripped = text.strip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise GraphScriptError("NON_JSON", str(exc)) from exc
    if stripped[end:].strip():
        raise GraphScriptError("EXTRA_TEXT", "output must contain exactly one JSON value")
    return value


def parse_graphscript(value: str | object, *, max_follow_limit: int = 100) -> GraphScript:
    raw = _decode_exact_json(value) if isinstance(value, str) else value
    try:
        script = GraphScript.model_validate(raw)
    except ValidationError as exc:
        raise GraphScriptError(_validation_reason(exc), str(exc)) from exc
    if script.version == "0.1":
        _validate_v01(script, max_follow_limit=max_follow_limit)
    else:
        _validate_v02(script, max_follow_limit=max_follow_limit)
    return script


def _validate_handle(handle: str) -> None:
    if HANDLE_PATTERN.fullmatch(handle) is None:
        raise GraphScriptError("INVALID_HANDLE", f"invalid handle: {handle!r}")


def _validate_v01(script: GraphScript, *, max_follow_limit: int) -> None:
    if len(script.ops) != 5:
        raise GraphScriptError("INVALID_SHAPE", "v0.1 requires exactly five operations")
    expected = (StartOp, FollowOp, FollowOp, RequireUniqueOp, EmitOp)
    if any(not isinstance(op, kind) for op, kind in zip(script.ops, expected, strict=True)):
        raise GraphScriptError(
            "INVALID_SHAPE", "expected start -> follow -> follow -> require_unique -> emit"
        )
    defined: set[str] = set()
    previous: str | None = None
    for op in script.ops:
        if isinstance(op, StartOp):
            _define_handle(op.out, defined)
            previous = op.out
        elif isinstance(op, FollowOp):
            _require_handle(op.input_handle, defined)
            _define_handle(op.out, defined)
            if op.input_handle != previous:
                raise GraphScriptError("INVALID_SHAPE", "v0.1 follows must form one chain")
            if op.limit > max_follow_limit:
                raise GraphScriptError(
                    "LIMIT_EXCEEDED", f"follow limit {op.limit} exceeds {max_follow_limit}"
                )
            previous = op.out
        elif isinstance(op, RequireUniqueOp | EmitOp):
            _require_handle(op.input_handle, defined)
            if op.input_handle != previous:
                raise GraphScriptError("INVALID_SHAPE", "unique/emit must consume final handle")


def _define_handle(handle: str, defined: set[str]) -> None:
    _validate_handle(handle)
    if handle in defined:
        raise GraphScriptError("DUPLICATE_HANDLE", f"duplicate handle: {handle}")
    defined.add(handle)


def _require_handle(handle: str, defined: set[str]) -> None:
    _validate_handle(handle)
    if handle not in defined:
        raise GraphScriptError("INVALID_HANDLE", f"undefined handle: {handle}")


def _validate_v02(script: GraphScript, *, max_follow_limit: int) -> None:
    if len(script.ops) < 2:
        raise GraphScriptError("INVALID_SHAPE", "v0.2 requires a root and emit")
    if not isinstance(script.ops[0], StartOp | AllEntitiesOp | ResolveEntityOp | SearchPassageOp):
        raise GraphScriptError(
            "INVALID_SHAPE",
            "v0.2 must start with start, all_entities, resolve_entity, or search_passage",
        )
    if not isinstance(script.ops[-1], EmitOp):
        raise GraphScriptError("INVALID_SHAPE", "v0.2 must end with emit")
    if sum(isinstance(op, EmitOp) for op in script.ops) != 1:
        raise GraphScriptError("INVALID_SHAPE", "v0.2 requires exactly one emit")

    defined: set[str] = set()
    kinds: dict[str, Literal["entity", "passage", "answer"]] = {}
    for op in script.ops:
        if isinstance(op, StartOp | AllEntitiesOp | ResolveEntityOp):
            _define_handle(op.out, defined)
            kinds[op.out] = "entity"
        elif isinstance(op, SearchPassageOp):
            _define_handle(op.out, defined)
            kinds[op.out] = "passage"
        elif isinstance(op, PassagePagesOp):
            _require_kind(op.input_handle, defined, kinds, "passage")
            _define_handle(op.out, defined)
            kinds[op.out] = "entity"
        elif isinstance(op, FollowOp):
            _require_kind(op.input_handle, defined, kinds, "entity")
            if op.limit > max_follow_limit:
                raise GraphScriptError(
                    "LIMIT_EXCEEDED", f"follow limit {op.limit} exceeds {max_follow_limit}"
                )
            _define_handle(op.out, defined)
            kinds[op.out] = "entity"
        elif isinstance(op, IntersectOp | UnionOp):
            for handle in op.inputs:
                _require_kind(handle, defined, kinds, "entity")
            _define_handle(op.out, defined)
            kinds[op.out] = "entity"
        elif isinstance(op, FilterTypeOp | FilterLiteralOp | SelectAmongOp):
            _require_kind(op.input_handle, defined, kinds, "entity")
            _define_handle(op.out, defined)
            kinds[op.out] = "entity"
        elif isinstance(op, CountOp | QueryAttributeOp):
            _require_kind(op.input_handle, defined, kinds, "entity")
            _define_handle(op.out, defined)
            kinds[op.out] = "answer"
        elif isinstance(op, QueryRelationOp):
            _require_kind(op.subject, defined, kinds, "entity")
            _require_kind(op.object, defined, kinds, "entity")
            _define_handle(op.out, defined)
            kinds[op.out] = "answer"
        elif isinstance(op, SelectBetweenOp):
            _require_kind(op.left, defined, kinds, "entity")
            _require_kind(op.right, defined, kinds, "entity")
            _define_handle(op.out, defined)
            kinds[op.out] = "entity"
        elif isinstance(op, RequireUniqueOp | EmitOp):
            _require_handle(op.input_handle, defined)
            if kinds[op.input_handle] == "passage":
                raise GraphScriptError(
                    "TYPE_MISMATCH", "passages must be converted with passage_pages before output"
                )


def _require_kind(
    handle: str,
    defined: set[str],
    kinds: dict[str, Literal["entity", "passage", "answer"]],
    expected: Literal["entity", "passage", "answer"],
) -> None:
    _require_handle(handle, defined)
    if kinds[handle] != expected:
        raise GraphScriptError(
            "TYPE_MISMATCH", f"handle {handle} is {kinds[handle]}, expected {expected}"
        )
