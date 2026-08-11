from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.schema.entity import AnswerSet, EntityInfo, Triple


class PassageHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_id: str = Field(min_length=1)
    paragraph_id: int = Field(ge=0)
    title: str
    text: str
    score: float


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["search", "text_search", "inspect_entity", "final_answer"]
    arguments: dict[str, Any]
    trace_id: str = Field(min_length=1)


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)
    step: int
    message: str
    triples: tuple[Triple, ...] = ()
    passages: tuple[PassageHit, ...] = ()
    entity: EntityInfo | None = None
    entities: tuple[EntityInfo, ...] = ()
    count: int | None = None
    values: tuple[str, ...] = ()
    answer_kind: Literal["entity", "literal", "count"] = "entity"
    total_entities: int = Field(default=0, ge=0)
    truncated: bool = False
    done: bool = False
    warnings: tuple[str, ...] = ()


class StepResult(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = {}


class EpisodeInput(BaseModel):
    task_id: str
    question: str
    topic_entity_ids: tuple[str, ...]
    gold_answers: AnswerSet


class Trajectory(BaseModel):
    task_id: str
    role: Literal["questioner", "solver"]
    seed: int
    calls: tuple[ToolCall, ...]
    observations: tuple[Observation, ...]
    final_answers: AnswerSet
    reward_components: dict[str, float] = {}
