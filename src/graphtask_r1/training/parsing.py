from __future__ import annotations

import json
import re
from typing import Any

from graphtask_r1.schema import AnswerSet, Program, parse_program


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


def parse_solver_output(text: str, *, count: bool = False) -> AnswerSet:
    payload = json.loads(_tag(text, "answer"))
    if not isinstance(payload, list):
        payload = [payload]
    if count:
        if len(payload) != 1:
            raise ValueError("count answer must contain exactly one value")
        return AnswerSet.count(int(payload[0]))
    return AnswerSet.entities([str(value) for value in payload])
