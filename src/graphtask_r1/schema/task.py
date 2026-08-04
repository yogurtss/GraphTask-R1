from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.schema.entity import AnswerSet, EntityInfo, Triple
from graphtask_r1.schema.program import Program


class VerificationSummary(BaseModel):
    executable: bool
    semantic_equivalent: bool | None = None
    necessity_mean: float = 0.0
    necessity_min: float = 0.0
    shortcut_found: bool | None = None
    answer_leak: bool = False


class TaskCertificate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    source: str
    question: str = Field(min_length=1)
    topic_entities: tuple[EntityInfo, ...]
    program: Program
    sparql: str
    gold_answers: AnswerSet
    witness_facts: tuple[Triple, ...]
    program_signature: str
    program_cost: float = Field(ge=0)
    operator_tags: tuple[str, ...] = ()
    verification: VerificationSummary
    solver_stats: dict[str, Any] = {}
    generation: dict[str, Any] = {}
