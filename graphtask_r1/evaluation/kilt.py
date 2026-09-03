from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.evaluation.answer_metrics import normalize_openqa_answer
from graphtask_r1.schema import AnswerSet, EvidenceProvenance


class KILTQAMetrics(BaseModel):
    """Answer and provenance metrics for one KILT QA prediction."""

    model_config = ConfigDict(frozen=True)

    answer_exact_match: float = Field(ge=0.0, le=1.0)
    answer_f1: float = Field(ge=0.0, le=1.0)
    provenance_rprecision: float = Field(ge=0.0, le=1.0)
    provenance_recall: float = Field(ge=0.0, le=1.0)
    kilt_exact_match: float = Field(ge=0.0, le=1.0)
    kilt_f1: float = Field(ge=0.0, le=1.0)


def kilt_qa_metrics(
    predicted_answer: AnswerSet,
    predicted_provenance: Sequence[EvidenceProvenance],
    answer_aliases: tuple[tuple[str, ...], ...],
    gold_provenance: tuple[tuple[EvidenceProvenance, ...], ...],
    *,
    recall_at: int = 5,
) -> KILTQAMetrics:
    """Score aliases and alternative complete evidence sets without using hidden answers."""

    predictions = tuple(str(answer.value) for answer in predicted_answer.answers)
    answer_exact, answer_f1 = _best_answer_score(predictions, answer_aliases)
    predicted_pages = tuple(dict.fromkeys(value.page_id for value in predicted_provenance))
    rprecision = _provenance_rprecision(predicted_pages, gold_provenance)
    recall = _provenance_recall(predicted_pages[:recall_at], gold_provenance)
    provenance_gate = float(rprecision == 1.0)
    return KILTQAMetrics(
        answer_exact_match=answer_exact,
        answer_f1=answer_f1,
        provenance_rprecision=rprecision,
        provenance_recall=recall,
        kilt_exact_match=answer_exact * provenance_gate,
        kilt_f1=answer_f1 * provenance_gate,
    )


def _best_answer_score(
    predictions: tuple[str, ...], aliases: tuple[tuple[str, ...], ...]
) -> tuple[float, float]:
    if not predictions and not aliases:
        return 1.0, 1.0
    if not predictions or not aliases:
        return 0.0, 0.0
    exact = 0.0
    f1 = 0.0
    for prediction in predictions:
        for group in aliases:
            for alias in group:
                exact = max(
                    exact,
                    float(normalize_openqa_answer(prediction) == normalize_openqa_answer(alias)),
                )
                f1 = max(f1, _token_f1(prediction, alias))
    return exact, f1


def _token_f1(prediction: str, gold: str) -> float:
    predicted_tokens = normalize_openqa_answer(prediction).split()
    gold_tokens = normalize_openqa_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _gold_page_sets(
    provenance: tuple[tuple[EvidenceProvenance, ...], ...],
) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(value.page_id for value in evidence_set) for evidence_set in provenance
    )


def _provenance_rprecision(
    predicted_pages: tuple[str, ...],
    gold_provenance: tuple[tuple[EvidenceProvenance, ...], ...],
) -> float:
    gold_sets = _gold_page_sets(gold_provenance)
    if not predicted_pages and not gold_sets:
        return 1.0
    if not gold_sets:
        return 0.0
    scores = []
    for gold_pages in gold_sets:
        if not gold_pages:
            continue
        top_r = set(predicted_pages[: len(gold_pages)])
        scores.append(len(top_r & gold_pages) / len(gold_pages))
    return max(scores, default=0.0)


def _provenance_recall(
    predicted_pages: tuple[str, ...],
    gold_provenance: tuple[tuple[EvidenceProvenance, ...], ...],
) -> float:
    gold_sets = _gold_page_sets(gold_provenance)
    if not predicted_pages and not gold_sets:
        return 1.0
    if not gold_sets:
        return 0.0
    predicted = set(predicted_pages)
    return max(
        (
            len(predicted & gold_pages) / len(gold_pages)
            for gold_pages in gold_sets
            if gold_pages
        ),
        default=0.0,
    )
