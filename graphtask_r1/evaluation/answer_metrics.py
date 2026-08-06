from __future__ import annotations

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
