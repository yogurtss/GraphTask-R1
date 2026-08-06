from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.schema.entity import AnswerSet, EntityInfo, Triple


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["search", "inspect_entity", "final_answer"]
    arguments: dict[str, Any]
    trace_id: str = Field(min_length=1)


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)
    step: int
    message: str
    triples: tuple[Triple, ...] = ()
    entity: EntityInfo | None = None
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
