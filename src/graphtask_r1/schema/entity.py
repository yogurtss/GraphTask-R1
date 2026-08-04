from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(min_length=1)
    label: str
    aliases: tuple[str, ...] = ()
    type_ids: tuple[str, ...] = ()


class RelationInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation_id: str = Field(min_length=1)
    label: str
    domain_types: tuple[str, ...] = ()
    range_types: tuple[str, ...] = ()


class Triple(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    object: str = Field(min_length=1)

    def sort_key(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str | int | float
    kind: Literal["entity", "literal", "count"] = "entity"
    label: str | None = None

    @field_validator("value")
    @classmethod
    def reject_empty(cls, value: str | int | float) -> str | int | float:
        if isinstance(value, str) and not value.strip():
            raise ValueError("answer value must not be empty")
        return value

    def sort_key(self) -> tuple[str, str]:
        return (self.kind, str(self.value))


class AnswerSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    answers: tuple[Answer, ...] = ()

    @field_validator("answers")
    @classmethod
    def canonicalize(cls, answers: tuple[Answer, ...]) -> tuple[Answer, ...]:
        unique = {(a.kind, str(a.value)): a for a in answers}
        return tuple(sorted(unique.values(), key=Answer.sort_key))

    @classmethod
    def entities(cls, values: set[str] | list[str] | tuple[str, ...]) -> AnswerSet:
        return cls(answers=tuple(Answer(value=value) for value in sorted(set(values))))

    @classmethod
    def count(cls, value: int) -> AnswerSet:
        return cls(answers=(Answer(value=value, kind="count"),))

    def values(self) -> tuple[str | int | float, ...]:
        return tuple(answer.value for answer in self.answers)

    def entity_ids(self) -> tuple[str, ...]:
        return tuple(str(answer.value) for answer in self.answers if answer.kind == "entity")


class Witness(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    facts: tuple[Triple, ...]
