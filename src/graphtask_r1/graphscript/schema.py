from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

HANDLE_PATTERN = re.compile(r"^h[0-7]$")


class GraphScriptError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class StartOp(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["start"] = "start"
    entity: Literal["$seed"] = "$seed"
    out: str


class FollowOp(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    op: Literal["follow"] = "follow"
    input_handle: str = Field(alias="in")
    relation: str = Field(min_length=1)
    direction: Literal["out", "in"] = "out"
    limit: int = Field(gt=0)
    out: str


class RequireUniqueOp(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    op: Literal["require_unique"] = "require_unique"
    input_handle: str = Field(alias="in")


class EmitOp(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    op: Literal["emit"] = "emit"
    input_handle: str = Field(alias="in")


GraphScriptOp: TypeAlias = Annotated[
    StartOp | FollowOp | RequireUniqueOp | EmitOp, Field(discriminator="op")
]
OP_ADAPTER: TypeAdapter[GraphScriptOp] = TypeAdapter(GraphScriptOp)


class GraphScript(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["0.1"] = "0.1"
    ops: tuple[GraphScriptOp, ...] = Field(min_length=1, max_length=5)


class BudgetUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_visits: int = Field(default=0, ge=0)
    operators: int = Field(default=0, ge=0)
    returned_entities: int = Field(default=0, ge=0)
    graph_calls: int = Field(default=0, ge=0)


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
    _validate_shape(script, max_follow_limit=max_follow_limit)
    return script


def _validate_handle(handle: str) -> None:
    if HANDLE_PATTERN.fullmatch(handle) is None:
        raise GraphScriptError("INVALID_HANDLE", f"invalid handle: {handle!r}")


def _validate_shape(script: GraphScript, *, max_follow_limit: int) -> None:
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
            _validate_handle(op.out)
            defined.add(op.out)
            previous = op.out
        elif isinstance(op, FollowOp):
            _validate_handle(op.input_handle)
            _validate_handle(op.out)
            if op.input_handle not in defined:
                raise GraphScriptError("INVALID_HANDLE", f"undefined handle: {op.input_handle}")
            if op.input_handle != previous:
                raise GraphScriptError("INVALID_SHAPE", "v0.1 follows must form one chain")
            if op.out in defined:
                raise GraphScriptError("DUPLICATE_HANDLE", f"duplicate handle: {op.out}")
            if op.limit > max_follow_limit:
                raise GraphScriptError(
                    "LIMIT_EXCEEDED", f"follow limit {op.limit} exceeds {max_follow_limit}"
                )
            defined.add(op.out)
            previous = op.out
        elif isinstance(op, RequireUniqueOp | EmitOp):
            _validate_handle(op.input_handle)
            if op.input_handle not in defined:
                raise GraphScriptError("INVALID_HANDLE", f"undefined handle: {op.input_handle}")
            if op.input_handle != previous:
                raise GraphScriptError("INVALID_SHAPE", "unique/emit must consume final handle")
