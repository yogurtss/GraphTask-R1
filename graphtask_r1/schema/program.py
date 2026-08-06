from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["entity"] = "entity"
    entity_id: str = Field(min_length=1)


class AllEntities(BaseModel):
    """Bounded universe scan used by supervised-data importers."""

    model_config = ConfigDict(frozen=True)
    op: Literal["all_entities"] = "all_entities"
    max_results: int = Field(default=10_000, gt=0, le=1_000_000)


class Hop(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["hop"] = "hop"
    input: Program
    relation: str = Field(min_length=1)
    direction: Literal["out", "in"] = "out"


class Intersect(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["intersect"] = "intersect"
    inputs: tuple[Program, ...]

    @field_validator("inputs")
    @classmethod
    def require_branches(cls, inputs: tuple[Program, ...]) -> tuple[Program, ...]:
        if len(inputs) < 2:
            raise ValueError("intersect requires at least two inputs")
        return inputs


class Union(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["union"] = "union"
    inputs: tuple[Program, ...]

    @field_validator("inputs")
    @classmethod
    def require_branches(cls, inputs: tuple[Program, ...]) -> tuple[Program, ...]:
        if len(inputs) < 2:
            raise ValueError("union requires at least two inputs")
        return inputs


class FilterType(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["filter_type"] = "filter_type"
    input: Program
    type_id: str = Field(min_length=1)


class LiteralValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str | int | float
    datatype: Literal["string", "quantity", "year", "date", "number"] = "string"
    unit: str | None = None


class FilterLiteral(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["filter_literal"] = "filter_literal"
    input: Program
    relation: str = Field(min_length=1)
    comparator: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains"] = "eq"
    value: LiteralValue

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: Any) -> Any:
        if isinstance(value, dict | LiteralValue):
            return value
        datatype = "number" if isinstance(value, int | float) else "string"
        return {"value": value, "datatype": datatype}


class Count(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["count"] = "count"
    input: Program


Program: TypeAlias = Annotated[
    AllEntities | Entity | Hop | Intersect | Union | FilterType | FilterLiteral | Count,
    Field(discriminator="op"),
]

Hop.model_rebuild()
Intersect.model_rebuild()
Union.model_rebuild()
FilterType.model_rebuild()
FilterLiteral.model_rebuild()
Count.model_rebuild()
PROGRAM_ADAPTER: TypeAdapter[Program] = TypeAdapter(Program)


def parse_program(value: object) -> Program:
    return PROGRAM_ADAPTER.validate_python(value)


def program_to_dict(program: Program) -> dict[str, object]:
    return cast(dict[str, object], PROGRAM_ADAPTER.dump_python(program, mode="json"))
