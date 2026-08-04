from __future__ import annotations

from typing import Any, Literal

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


class TaskProvenance(BaseModel):
    dataset: str | None = None
    raw_file: str | None = None
    raw_index: int | None = None
    converter_version: str | None = None
    source_hash: str | None = None


class TaskProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_entities: tuple[str, ...]
    program: Program
    paraphrase: str | None = None


class BenchmarkExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    dataset: Literal["webqsp", "cwq", "grailqa"]
    split: str
    question: str
    topic_entity_ids: tuple[str, ...]
    gold_answers: AnswerSet
    logical_form: str | None = None
    sparql: str | None = None
    metadata: dict[str, Any] = {}


class TaskCertificate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    task_id: str = Field(min_length=1)
    source: str
    source_id: str | None = None
    split: str | None = None
    graph_snapshot: str = "toy-v1"
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
    source_program: dict[str, Any] | None = None
    provenance: TaskProvenance = TaskProvenance()
    solver_stats: dict[str, Any] = {}
    generation: dict[str, Any] = {}
