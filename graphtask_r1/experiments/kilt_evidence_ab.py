from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.evaluation import KILTQAMetrics, kilt_qa_metrics
from graphtask_r1.graphscript.evidence_flow import EvidenceRetriever
from graphtask_r1.schema import AnswerSet, BenchmarkExample, EvidenceProvenance, PassageHit


class AnswerPrediction(BaseModel):
    """Structured output shared by the baseline and Evidence-Flow policy."""

    model_config = ConfigDict(frozen=True)

    answer: AnswerSet
    passage_keys: tuple[str, ...] = ()
    rejection_reason: str | None = None


class EvidenceAnswerer(Protocol):
    def answer(
        self,
        question: str,
        passages: tuple[PassageHit, ...],
        *,
        trace_id: str,
    ) -> AnswerPrediction: ...


class EvidenceABConfig(BaseModel):
    """A controlled small-scale comparison with a fixed context budget."""

    model_config = ConfigDict(frozen=True)

    retrieve_k: int = Field(default=3, ge=1, le=10)
    expand_k: int = Field(default=3, ge=1, le=10)
    context_k: int = Field(default=5, ge=1, le=20)
    recall_at: int = Field(default=5, ge=1, le=100)


class EvidencePolicyRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["single_retrieval", "evidence_flow"]
    example_id: str
    prediction: AnswerPrediction
    provenance: tuple[EvidenceProvenance, ...]
    observed_passages: tuple[PassageHit, ...]
    search_calls: int = Field(ge=1)


class EvidenceABExampleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    example_id: str
    baseline_prediction: AnswerPrediction
    evidence_flow_prediction: AnswerPrediction
    baseline: KILTQAMetrics
    evidence_flow: KILTQAMetrics


class EvidenceABReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    examples: int = Field(ge=1)
    baseline: dict[str, float]
    evidence_flow: dict[str, float]
    delta: dict[str, float]
    per_example: tuple[EvidenceABExampleResult, ...]


class SingleRetrievalBaseline:
    """One BM25 retrieval followed by one answer call."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        answerer: EvidenceAnswerer,
        config: EvidenceABConfig,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.config = config

    def run(self, example: BenchmarkExample) -> EvidencePolicyRun:
        passages = self.retriever.retrieve(
            example.question,
            limit=self.config.context_k,
            trace_id=f"ab:{example.example_id}:baseline:retrieve",
        )
        prediction = self.answerer.answer(
            example.question,
            passages,
            trace_id=f"ab:{example.example_id}:baseline:answer",
        )
        provenance = _selected_provenance(prediction, passages)
        return EvidencePolicyRun(
            method="single_retrieval",
            example_id=example.example_id,
            prediction=prediction,
            provenance=provenance,
            observed_passages=passages,
            search_calls=1,
        )


class HyperlinkEvidenceFlow:
    """Retrieve bridge pages, expand their links, then answer over a fixed context budget."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        answerer: EvidenceAnswerer,
        config: EvidenceABConfig,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.config = config

    def run(self, example: BenchmarkExample) -> EvidencePolicyRun:
        first_hop = self.retriever.retrieve(
            example.question,
            limit=self.config.retrieve_k,
            trace_id=f"ab:{example.example_id}:flow:retrieve",
        )
        second_hop = self.retriever.expand(
            first_hop,
            example.question,
            limit=self.config.expand_k,
            trace_id=f"ab:{example.example_id}:flow:expand",
        )
        passages = _merge_passages(first_hop, second_hop)[: self.config.context_k]
        prediction = self.answerer.answer(
            example.question,
            passages,
            trace_id=f"ab:{example.example_id}:flow:answer",
        )
        provenance = _selected_provenance(prediction, passages)
        return EvidencePolicyRun(
            method="evidence_flow",
            example_id=example.example_id,
            prediction=prediction,
            provenance=provenance,
            observed_passages=passages,
            search_calls=2,
        )


def benchmark_examples_from_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    example_ids: frozenset[str] | None = None,
    limit: int | None = None,
) -> tuple[BenchmarkExample, ...]:
    """Read either benchmark rows or nested EvidenceSelfPlayRecord audit rows."""

    examples: list[BenchmarkExample] = []
    for row in rows:
        raw_example: Mapping[str, Any] = row
        challenge = row.get("challenge")
        if isinstance(challenge, Mapping):
            nested = challenge.get("example")
            if isinstance(nested, Mapping):
                raw_example = nested
        example = BenchmarkExample.model_validate(raw_example)
        if example_ids is not None and example.example_id not in example_ids:
            continue
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break
    return tuple(examples)


def evaluate_evidence_ab(
    examples: Sequence[BenchmarkExample],
    baseline: SingleRetrievalBaseline,
    evidence_flow: HyperlinkEvidenceFlow,
) -> EvidenceABReport:
    """Run a paired comparison so both policies see exactly the same examples."""

    if not examples:
        raise ValueError("at least one KILT example is required")
    rows: list[EvidenceABExampleResult] = []
    for example in examples:
        baseline_run = baseline.run(example)
        flow_run = evidence_flow.run(example)
        baseline_metrics = _score(example, baseline_run, baseline.config.recall_at)
        flow_metrics = _score(example, flow_run, evidence_flow.config.recall_at)
        rows.append(
            EvidenceABExampleResult(
                example_id=example.example_id,
                baseline_prediction=baseline_run.prediction,
                evidence_flow_prediction=flow_run.prediction,
                baseline=baseline_metrics,
                evidence_flow=flow_metrics,
            )
        )
    baseline_summary = _mean_metrics(tuple(row.baseline for row in rows))
    flow_summary = _mean_metrics(tuple(row.evidence_flow for row in rows))
    return EvidenceABReport(
        examples=len(rows),
        baseline=baseline_summary,
        evidence_flow=flow_summary,
        delta={key: flow_summary[key] - baseline_summary[key] for key in baseline_summary},
        per_example=tuple(rows),
    )


def _score(
    example: BenchmarkExample, run: EvidencePolicyRun, recall_at: int
) -> KILTQAMetrics:
    aliases = example.answer_aliases
    if not aliases:
        aliases = tuple((str(answer.value),) for answer in example.gold_answers.answers)
    return kilt_qa_metrics(
        run.prediction.answer,
        run.provenance,
        aliases,
        example.gold_provenance,
        recall_at=recall_at,
    )


def _merge_passages(*groups: tuple[PassageHit, ...]) -> tuple[PassageHit, ...]:
    merged: dict[str, PassageHit] = {}
    for group in groups:
        for passage in group:
            merged.setdefault(_passage_key(passage), passage)
    return tuple(merged.values())


def _selected_provenance(
    prediction: AnswerPrediction, passages: tuple[PassageHit, ...]
) -> tuple[EvidenceProvenance, ...]:
    candidates = {_passage_key(passage): passage for passage in passages}
    keys = prediction.passage_keys or tuple(candidates)
    return tuple(
        EvidenceProvenance(
            page_id=candidates[key].page_id,
            paragraph_id=candidates[key].paragraph_id,
            title=candidates[key].title,
        )
        for key in dict.fromkeys(keys)
        if key in candidates
    )


def _passage_key(passage: PassageHit) -> str:
    return f"{passage.page_id}:{passage.paragraph_id}"


def _mean_metrics(values: tuple[KILTQAMetrics, ...]) -> dict[str, float]:
    fields = tuple(KILTQAMetrics.model_fields)
    return {
        field: sum(float(getattr(value, field)) for value in values) / len(values)
        for field in fields
    }
