from __future__ import annotations

from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    AllEntities,
    Count,
    Entity,
    EntityInfo,
    FilterLiteral,
    FilterQualifier,
    FilterType,
    Hop,
    LiteralValue,
    Program,
    QueryAttribute,
    QueryAttributeQualifier,
    QueryAttributeUnderCondition,
    QueryRelation,
    QueryRelationQualifier,
    SelectAmong,
    Union,
    Verify,
)

MAX_COMPACT_QUERY_STEPS = 32
MAX_COMPACT_QUERY_ENTITIES = 4_096


class EntityQueryRoot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["entities"] = "entities"
    entity_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_COMPACT_QUERY_ENTITIES)

    @field_validator("entity_ids")
    @classmethod
    def canonicalize_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("query entity IDs cannot be empty")
        return tuple(sorted(set(values)))


class AllEntitiesQueryRoot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["all_entities"] = "all_entities"
    max_results: int = Field(default=1_000_000, gt=0, le=1_000_000)


QueryRoot = Annotated[EntityQueryRoot | AllEntitiesQueryRoot, Field(discriminator="kind")]


class HopQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["hop"] = "hop"
    relation: str = Field(min_length=1)
    direction: Literal["out", "in"] = "out"


class FilterTypeQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["filter_type"] = "filter_type"
    type_ids: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator("type_ids")
    @classmethod
    def canonicalize_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("query type IDs cannot be empty")
        return tuple(sorted(set(values)))


class FilterLiteralQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["filter_literal"] = "filter_literal"
    relation: str = Field(min_length=1)
    comparator: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains"] = "eq"
    value: LiteralValue


class FilterQualifierQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["filter_qualifier"] = "filter_qualifier"
    qualifier: str = Field(min_length=1)
    comparator: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains"] = "eq"
    value: LiteralValue


class QueryAttributeQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["query_attribute"] = "query_attribute"
    attribute: str = Field(min_length=1)


class QueryAttributeUnderConditionQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["query_attribute_under_condition"] = "query_attribute_under_condition"
    attribute: str = Field(min_length=1)
    qualifier: str = Field(min_length=1)
    qualifier_value: LiteralValue


class QueryAttributeQualifierQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["query_attribute_qualifier"] = "query_attribute_qualifier"
    attribute: str = Field(min_length=1)
    attribute_value: LiteralValue
    qualifier: str = Field(min_length=1)


class QueryRelationQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["query_relation"] = "query_relation"
    object_entity_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_COMPACT_QUERY_ENTITIES)


class QueryRelationQualifierQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["query_relation_qualifier"] = "query_relation_qualifier"
    object_entity_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_COMPACT_QUERY_ENTITIES)
    relation: str = Field(min_length=1)
    qualifier: str = Field(min_length=1)


class VerifyQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["verify"] = "verify"
    comparator: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains"] = "eq"
    value: LiteralValue


class SelectAttributeQueryStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["select_attribute"] = "select_attribute"
    attribute: str = Field(min_length=1)
    mode: Literal["min", "max"]


QueryStep = Annotated[
    HopQueryStep
    | FilterTypeQueryStep
    | FilterLiteralQueryStep
    | FilterQualifierQueryStep
    | QueryAttributeQueryStep
    | QueryAttributeUnderConditionQueryStep
    | QueryAttributeQualifierQueryStep
    | QueryRelationQueryStep
    | QueryRelationQualifierQueryStep
    | VerifyQueryStep
    | SelectAttributeQueryStep,
    Field(discriminator="op"),
]
QUERY_STEP_ADAPTER: TypeAdapter[QueryStep] = TypeAdapter(QueryStep)


class CompactGraphQuery(BaseModel):
    """Bounded, replayable server-side graph filtering without giant entity-ID prompts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: QueryRoot
    steps: tuple[QueryStep, ...] = Field(max_length=MAX_COMPACT_QUERY_STEPS)
    return_count: bool = False
    limit: int = Field(default=200, gt=0, le=MAX_COMPACT_QUERY_ENTITIES)

    @model_validator(mode="after")
    def require_restricted_global_query(self) -> CompactGraphQuery:
        if isinstance(self.root, AllEntitiesQueryRoot) and not self.steps:
            raise ValueError("all_entities compact queries require at least one restricting step")
        terminal = tuple(
            index
            for index, step in enumerate(self.steps)
            if isinstance(
                step,
                QueryAttributeQueryStep
                | QueryAttributeUnderConditionQueryStep
                | QueryAttributeQualifierQueryStep
                | QueryRelationQueryStep
                | QueryRelationQualifierQueryStep
                | SelectAttributeQueryStep,
            )
        )
        verify = tuple(
            index for index, step in enumerate(self.steps) if isinstance(step, VerifyQueryStep)
        )
        valid_terminal = (len(self.steps) - 1,)
        valid_verified = (len(self.steps) - 2,) if verify == (len(self.steps) - 1,) else ()
        if terminal and terminal not in {valid_terminal, valid_verified}:
            raise ValueError("query/select steps must appear exactly once and be terminal")
        if verify and (verify != (len(self.steps) - 1,) or terminal != valid_verified):
            raise ValueError("verify must follow one literal query as the terminal step")
        if self.return_count and (terminal or verify):
            raise ValueError("return_count cannot be combined with a query/select terminal step")
        return self


class CompactGraphQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: tuple[EntityInfo, ...] = ()
    count: int | None = None
    values: tuple[str, ...] = ()
    answer_kind: Literal["entity", "literal", "count"] = "entity"
    total_entities: int = Field(default=0, ge=0)
    truncated: bool = False


@runtime_checkable
class _BulkEntityInfoBackend(Protocol):
    def entity_infos(self, entity_ids: list[str]) -> tuple[EntityInfo, ...]: ...


def compact_query_to_program(query: CompactGraphQuery) -> Program:
    root = query.root
    if isinstance(root, EntityQueryRoot):
        entities = tuple(Entity(entity_id=value) for value in root.entity_ids)
        program: Program = entities[0] if len(entities) == 1 else Union(inputs=entities)
    else:
        program = AllEntities(max_results=root.max_results)

    for step in query.steps:
        if isinstance(step, HopQueryStep):
            program = Hop(
                input=program,
                relation=step.relation,
                direction=step.direction,
            )
        elif isinstance(step, FilterTypeQueryStep):
            branches = tuple(FilterType(input=program, type_id=value) for value in step.type_ids)
            program = branches[0] if len(branches) == 1 else Union(inputs=branches)
        elif isinstance(step, FilterLiteralQueryStep):
            program = FilterLiteral(
                input=program,
                relation=step.relation,
                comparator=step.comparator,
                value=step.value,
            )
        elif isinstance(step, FilterQualifierQueryStep):
            program = FilterQualifier(
                input=program,
                qualifier=step.qualifier,
                comparator=step.comparator,
                value=step.value,
            )
        elif isinstance(step, QueryAttributeQueryStep):
            program = QueryAttribute(input=program, attribute=step.attribute)
        elif isinstance(step, QueryAttributeUnderConditionQueryStep):
            program = QueryAttributeUnderCondition(
                input=program,
                attribute=step.attribute,
                qualifier=step.qualifier,
                qualifier_value=step.qualifier_value,
            )
        elif isinstance(step, QueryAttributeQualifierQueryStep):
            program = QueryAttributeQualifier(
                input=program,
                attribute=step.attribute,
                attribute_value=step.attribute_value,
                qualifier=step.qualifier,
            )
        elif isinstance(step, QueryRelationQueryStep):
            objects = tuple(Entity(entity_id=value) for value in step.object_entity_ids)
            object_program: Program = objects[0] if len(objects) == 1 else Union(inputs=objects)
            program = QueryRelation(subject=program, object=object_program)
        elif isinstance(step, QueryRelationQualifierQueryStep):
            objects = tuple(Entity(entity_id=value) for value in step.object_entity_ids)
            object_program = objects[0] if len(objects) == 1 else Union(inputs=objects)
            program = QueryRelationQualifier(
                subject=program,
                object=object_program,
                relation=step.relation,
                qualifier=step.qualifier,
            )
        elif isinstance(step, SelectAttributeQueryStep):
            program = SelectAmong(
                input=program,
                attribute=step.attribute,
                mode=step.mode,
            )
        elif isinstance(step, VerifyQueryStep):
            program = Verify(input=program, comparator=step.comparator, value=step.value)
        else:  # pragma: no cover - the discriminated union prevents this
            raise TypeError(type(step).__name__)
    return Count(input=program) if query.return_count else program


def execute_compact_query(
    backend: GraphBackend,
    value: object,
    *,
    max_limit: int = MAX_COMPACT_QUERY_ENTITIES,
) -> CompactGraphQueryResult:
    query = CompactGraphQuery.model_validate(value)
    effective_limit = min(query.limit, max(1, max_limit), MAX_COMPACT_QUERY_ENTITIES)
    answers = backend.execute_program(compact_query_to_program(query))
    if query.return_count:
        count = int(answers.values()[0]) if answers.answers else 0
        return CompactGraphQueryResult(count=count, answer_kind="count")

    if answers.answers and answers.answers[0].kind == "literal":
        values = tuple(str(answer.value) for answer in answers.answers)
        return CompactGraphQueryResult(
            values=values[:effective_limit],
            answer_kind="literal",
            total_entities=len(values),
            truncated=len(values) > effective_limit,
        )

    entity_ids = answers.entity_ids()
    selected = list(entity_ids[:effective_limit])
    if isinstance(backend, _BulkEntityInfoBackend):
        entities = backend.entity_infos(selected)
    else:
        entities = tuple(backend.entity_info(entity_id) for entity_id in selected)
    return CompactGraphQueryResult(
        entities=entities,
        answer_kind="entity",
        total_entities=len(entity_ids),
        truncated=len(entity_ids) > effective_limit,
    )


COMPACT_GRAPH_QUERY_SCHEMA: dict[str, object] = {
    "type": "object",
    "description": (
        "A bounded server-side traversal/filter query. Use this instead of listing "
        "large candidate sets."
    ),
    "properties": {
        "root": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "entities"},
                        "entity_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 512,
                        },
                    },
                    "required": ["kind", "entity_ids"],
                },
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "all_entities"},
                        "max_results": {"type": "integer", "maximum": 1000000},
                    },
                    "required": ["kind"],
                },
            ]
        },
        "steps": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "hop"},
                            "relation": {"type": "string"},
                            "direction": {"type": "string", "enum": ["out", "in"]},
                        },
                        "required": ["op", "relation"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "filter_type"},
                            "type_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 32,
                            },
                        },
                        "required": ["op", "type_ids"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "filter_literal"},
                            "relation": {"type": "string"},
                            "comparator": {
                                "type": "string",
                                "enum": ["eq", "ne", "lt", "le", "gt", "ge", "contains"],
                            },
                            "value": {"type": "object"},
                        },
                        "required": ["op", "relation", "value"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "query_attribute"},
                            "attribute": {"type": "string"},
                        },
                        "required": ["op", "attribute"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "query_relation"},
                            "object_entity_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 512,
                            },
                        },
                        "required": ["op", "object_entity_ids"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "select_attribute"},
                            "attribute": {"type": "string"},
                            "mode": {"type": "string", "enum": ["min", "max"]},
                        },
                        "required": ["op", "attribute", "mode"],
                    },
                ]
            },
        },
        "return_count": {"type": "boolean"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 512},
    },
    "required": ["root", "steps"],
}
