from __future__ import annotations

import re

from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import AnswerSet


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold()))


def answer_leak(question: str, answers: AnswerSet, backend: GraphBackend) -> bool:
    normalized = f" {normalize_text(question)} "
    for answer in answers.answers:
        if answer.kind != "entity":
            continue
        info = backend.entity_info(str(answer.value))
        for alias in (info.label, *info.aliases):
            candidate = normalize_text(alias)
            if candidate and f" {candidate} " in normalized:
                return True
    return False
