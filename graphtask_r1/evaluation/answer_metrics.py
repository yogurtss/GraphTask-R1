from __future__ import annotations

import re
import string

from graphtask_r1.schema import AnswerSet


def answer_metrics(predicted: AnswerSet, gold: AnswerSet) -> dict[str, float]:
    pred = {(answer.kind, str(answer.value)) for answer in predicted.answers}
    target = {(answer.kind, str(answer.value)) for answer in gold.answers}
    if not pred and not target:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    precision = len(pred & target) / len(pred) if pred else 0.0
    recall = len(pred & target) / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": float(pred == target),
    }


def normalize_openqa_answer(value: str) -> str:
    """Apply the normalized-EM convention used by Search-R1/CoEvoKG QA evaluation."""

    lowered = value.casefold()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def openqa_alias_metrics(
    predicted: AnswerSet, alias_groups: tuple[tuple[str, ...], ...]
) -> dict[str, float]:
    """Score open-domain QA answers, where aliases are alternatives rather than a target set."""

    predictions = {
        normalize_openqa_answer(str(answer.value))
        for answer in predicted.answers
        if normalize_openqa_answer(str(answer.value))
    }
    groups = tuple(
        {normalize_openqa_answer(alias) for alias in aliases if normalize_openqa_answer(alias)}
        for aliases in alias_groups
    )
    groups = tuple(group for group in groups if group)
    if not predictions and not groups:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    matched_predictions = {
        prediction for prediction in predictions if any(prediction in group for group in groups)
    }
    matched_groups = sum(bool(predictions & group) for group in groups)
    precision = len(matched_predictions) / len(predictions) if predictions else 0.0
    recall = matched_groups / len(groups) if groups else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = float(
        len(predictions) == len(groups)
        and len(matched_predictions) == len(predictions)
        and matched_groups == len(groups)
    )
    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact_match}
