from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from graphtask_r1.graphscript import GraphScript, GraphScriptError, parse_graphscript
from graphtask_r1.schema import AnswerSet, Program, TaskProposal, parse_program


def _tag(text: str, name: str) -> str:
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"missing <{name}> block")
    return match.group(1)


def parse_questioner_output(text: str) -> tuple[str, tuple[str, ...], Program]:
    payload: dict[str, Any] = json.loads(_tag(text, "task"))
    question = str(payload["question"]).strip()
    if not question:
        raise ValueError("question is empty")
    topic_entities = tuple(str(value) for value in payload["topic_entities"])
    return question, topic_entities, parse_program(payload["program"])


def parse_task_proposal(text: str) -> TaskProposal:
    payload = decode_task_proposal_output(text)
    if "question" in payload and "paraphrase" not in payload:
        payload["paraphrase"] = payload.pop("question")
    return TaskProposal.model_validate(payload)


def decode_task_proposal_output(text: str) -> dict[str, Any]:
    payload = json.loads(_tag(text, "task"))
    if not isinstance(payload, dict):
        raise ValueError("task payload must be an object")
    return dict(payload)


@dataclass(frozen=True)
class QuestionerGraphScriptPayload:
    """Decoded v3 Questioner output before GraphScript schema validation."""

    question: str | None
    program: object
    wrapped: bool


def decode_questioner_graphscript_output(text: str) -> QuestionerGraphScriptPayload:
    """Accept the v3 question/program envelope and legacy code-only GraphScript JSON."""

    payload: object = json.loads(text)
    if isinstance(payload, dict) and ("question" in payload or "program" in payload):
        question_value = payload.get("question")
        question = question_value.strip() if isinstance(question_value, str) else None
        return QuestionerGraphScriptPayload(
            question=question or None,
            program=payload.get("program"),
            wrapped=True,
        )
    return QuestionerGraphScriptPayload(question=None, program=payload, wrapped=False)


def parse_questioner_graphscript_output(
    text: str, *, max_follow_limit: int = 100
) -> tuple[str | None, GraphScript]:
    """Parse a Questioner envelope while retaining code-only rollout compatibility."""

    payload = decode_questioner_graphscript_output(text)
    if payload.wrapped and payload.question is None:
        raise GraphScriptError("INVALID_OUTPUT", "question must be a non-empty string")
    if payload.wrapped and payload.program is None:
        raise GraphScriptError("INVALID_OUTPUT", "program is required")
    return payload.question, parse_graphscript(
        payload.program,
        max_follow_limit=max_follow_limit,
    )


def parse_solver_output(
    text: str,
    *,
    count: bool = False,
    answer_kind: Literal["entity", "literal", "count"] | None = None,
) -> AnswerSet:
    payload = json.loads(_tag(text, "answer"))
    if not isinstance(payload, list):
        payload = [payload]
    kind = "count" if count else (answer_kind or "entity")
    if kind == "count":
        if len(payload) != 1:
            raise ValueError("count answer must contain exactly one value")
        return AnswerSet.count(int(payload[0]))
    if kind == "literal":
        return AnswerSet.literals([str(value) for value in payload])
    return AnswerSet.entities([str(value) for value in payload])
