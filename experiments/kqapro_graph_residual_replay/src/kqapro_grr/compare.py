from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.utils import file_hash, iter_record_json, stable_hash, write_json


class RejectionReason(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str | None = None


class PredictionRecord(BaseModel):
    """Fields needed for a paired GRR promotion decision."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    exact_match: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    tool_succeeded: bool
    rejection_reason: RejectionReason | None = None

    @property
    def correct(self) -> bool:
        return self.exact_match == 1.0

    @property
    def request_failed(self) -> bool:
        return (
            self.rejection_reason is not None
            and self.rejection_reason.code == "MODEL_REQUEST_FAILED"
        )


@dataclass(frozen=True)
class PredictionPair:
    baseline: PredictionRecord
    candidate: PredictionRecord


def _load_predictions(path: Path) -> list[PredictionRecord]:
    return [PredictionRecord.model_validate_json(raw) for raw in iter_record_json(path)]


def _index_predictions(
    records: Sequence[PredictionRecord], *, side: str
) -> dict[str, PredictionRecord]:
    counts = Counter(record.task_id for record in records)
    duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{side} predictions contain duplicate task_id: {duplicates[0]!r}")
    return {record.task_id: record for record in records}


def _pair_predictions(
    baseline: Sequence[PredictionRecord], candidate: Sequence[PredictionRecord]
) -> list[PredictionPair]:
    baseline_by_id = _index_predictions(baseline, side="baseline")
    candidate_by_id = _index_predictions(candidate, side="candidate")
    baseline_ids = set(baseline_by_id)
    candidate_ids = set(candidate_by_id)
    if baseline_ids != candidate_ids:
        baseline_only = sorted(baseline_ids - candidate_ids)
        candidate_only = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "prediction task_id sets differ: "
            f"baseline_only={baseline_only[:3]!r}, candidate_only={candidate_only[:3]!r}; "
            f"counts={len(baseline_ids)}/{len(candidate_ids)}"
        )
    return [
        PredictionPair(baseline=record, candidate=candidate_by_id[record.task_id])
        for record in baseline
    ]


def _metrics(records: Sequence[PredictionRecord]) -> dict[str, int | float]:
    count = len(records)
    correct_count = sum(record.correct for record in records)
    tool_success_count = sum(record.tool_succeeded for record in records)
    request_failure_count = sum(record.request_failed for record in records)
    return {
        "count": count,
        "correct_count": correct_count,
        "exact_match": correct_count / count,
        "f1": sum(record.f1 for record in records) / count,
        "tool_success_count": tool_success_count,
        "tool_success_rate": tool_success_count / count,
        "request_failure_count": request_failure_count,
    }


def exact_mcnemar_p(candidate_only: int, baseline_only: int) -> float:
    """Return the exact two-sided McNemar p-value under Binomial(n, 0.5)."""

    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    lower = min(candidate_only, baseline_only)
    lower_tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, float(2 * lower_tail / (2**discordant)))


def compare_predictions(
    baseline_path: Path,
    candidate_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Compare two evaluations on exactly the same task set."""

    baseline = _load_predictions(baseline_path)
    candidate = _load_predictions(candidate_path)
    pairs = _pair_predictions(baseline, candidate)
    if not pairs:
        raise ValueError("prediction files must contain at least one task")

    baseline_metrics = _metrics([pair.baseline for pair in pairs])
    candidate_metrics = _metrics([pair.candidate for pair in pairs])
    candidate_only_ids = [
        pair.candidate.task_id
        for pair in pairs
        if pair.candidate.correct and not pair.baseline.correct
    ]
    baseline_only_ids = [
        pair.baseline.task_id
        for pair in pairs
        if pair.baseline.correct and not pair.candidate.correct
    ]
    criteria = {
        "candidate_correct_count_strictly_higher": (
            candidate_metrics["correct_count"] > baseline_metrics["correct_count"]
        ),
        "mean_f1_non_decreasing": candidate_metrics["f1"] >= baseline_metrics["f1"],
        "tool_success_rate_non_decreasing": (
            candidate_metrics["tool_success_rate"] >= baseline_metrics["tool_success_rate"]
        ),
        "baseline_request_failures_zero": baseline_metrics["request_failure_count"] == 0,
        "candidate_request_failures_zero": candidate_metrics["request_failure_count"] == 0,
    }
    task_ids = [pair.baseline.task_id for pair in pairs]
    result: dict[str, object] = {
        "inputs": {
            "baseline_path": str(baseline_path),
            "candidate_path": str(candidate_path),
            "baseline_sha256": file_hash(baseline_path),
            "candidate_sha256": file_hash(candidate_path),
            "task_set_sha256": stable_hash(sorted(task_ids)),
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": {
            "correct_count": candidate_metrics["correct_count"]
            - baseline_metrics["correct_count"],
            "exact_match": candidate_metrics["exact_match"] - baseline_metrics["exact_match"],
            "f1": candidate_metrics["f1"] - baseline_metrics["f1"],
            "tool_success_rate": candidate_metrics["tool_success_rate"]
            - baseline_metrics["tool_success_rate"],
            "request_failure_count": candidate_metrics["request_failure_count"]
            - baseline_metrics["request_failure_count"],
        },
        "discordance": {
            "candidate_only_count": len(candidate_only_ids),
            "candidate_only_task_ids": candidate_only_ids,
            "baseline_only_count": len(baseline_only_ids),
            "baseline_only_task_ids": baseline_only_ids,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(
                len(candidate_only_ids), len(baseline_only_ids)
            ),
        },
        "gate": {"passed": all(criteria.values()), "criteria": criteria},
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


__all__ = ["compare_predictions", "exact_mcnemar_p"]
