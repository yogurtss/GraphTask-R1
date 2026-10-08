from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.utils import file_hash, iter_record_json, stable_hash, write_json


class RejectionReason(BaseModel):
    """The structured evaluator fields used for failure diagnosis."""

    model_config = ConfigDict(extra="ignore")

    code: str | None = None
    cause_code: str | None = None


class PredictionRecord(BaseModel):
    """Comparable fields from one KQAPro prediction record."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    exact_match: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    tool_succeeded: bool
    rejection_reason: RejectionReason | None = None
    operator_tags: tuple[str, ...] = ()

    @property
    def correct(self) -> bool:
        return self.exact_match == 1.0

    @property
    def model_request_failed(self) -> bool:
        return (
            self.rejection_reason is not None
            and self.rejection_reason.code == "MODEL_REQUEST_FAILED"
        )

    @property
    def diagnostic_reason(self) -> str:
        if self.correct:
            return "EXACT_MATCH"
        if self.rejection_reason is not None:
            if self.rejection_reason.cause_code:
                return self.rejection_reason.cause_code
            if self.rejection_reason.code:
                return self.rejection_reason.code
        if self.tool_succeeded:
            return "TOOL_WRONG_ANSWER"
        return "DIRECT_WRONG_ANSWER"


@dataclass(frozen=True)
class PredictionPair:
    baseline: PredictionRecord
    candidate: PredictionRecord


Side = Literal["baseline", "candidate"]
Metric = Literal["exact_match", "f1", "tool_success_rate"]


def _load_predictions(path: Path) -> list[PredictionRecord]:
    return [PredictionRecord.model_validate_json(raw) for raw in iter_record_json(path)]


def _pair_predictions(
    baseline: Sequence[PredictionRecord], candidate: Sequence[PredictionRecord]
) -> list[PredictionPair]:
    baseline_ids = [record.task_id for record in baseline]
    candidate_ids = [record.task_id for record in candidate]
    for side, task_ids in (("baseline", baseline_ids), ("candidate", candidate_ids)):
        duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"{side} predictions contain duplicate task_id: {duplicates[0]!r}")
    if baseline_ids != candidate_ids:
        mismatch = next(
            (
                index
                for index, (baseline_id, candidate_id) in enumerate(
                    zip(baseline_ids, candidate_ids, strict=False)
                )
                if baseline_id != candidate_id
            ),
            min(len(baseline_ids), len(candidate_ids)),
        )
        baseline_id = baseline_ids[mismatch] if mismatch < len(baseline_ids) else "<missing>"
        candidate_id = candidate_ids[mismatch] if mismatch < len(candidate_ids) else "<missing>"
        raise ValueError(
            "ordered task_id mismatch at index "
            f"{mismatch}: baseline={baseline_id!r}, candidate={candidate_id!r}; "
            f"counts={len(baseline_ids)}/{len(candidate_ids)}"
        )
    return [
        PredictionPair(baseline=baseline_record, candidate=candidate_record)
        for baseline_record, candidate_record in zip(baseline, candidate, strict=True)
    ]


def _record(pair: PredictionPair, side: Side) -> PredictionRecord:
    return pair.baseline if side == "baseline" else pair.candidate


def _metrics(pairs: Sequence[PredictionPair], side: Side) -> dict[str, int | float]:
    count = len(pairs)
    correct_count = sum(_record(pair, side).correct for pair in pairs)
    tool_success_count = sum(_record(pair, side).tool_succeeded for pair in pairs)
    model_request_failed_count = sum(_record(pair, side).model_request_failed for pair in pairs)
    return {
        "count": count,
        "correct_count": correct_count,
        "exact_match": correct_count / count,
        "f1": sum(_record(pair, side).f1 for pair in pairs) / count,
        "tool_success_count": tool_success_count,
        "tool_success_rate": tool_success_count / count,
        "model_request_failed_count": model_request_failed_count,
    }


def exact_mcnemar_p(candidate_only: int, baseline_only: int) -> float:
    """Return the exact two-sided McNemar p-value via a Binomial(n, 0.5) tail."""

    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    lower = min(candidate_only, baseline_only)
    tail_numerator = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, float(2 * tail_numerator / (2**discordant)))


def _metric_difference(pair: PredictionPair, metric: Metric) -> float:
    if metric == "exact_match":
        return float(pair.candidate.correct) - float(pair.baseline.correct)
    if metric == "f1":
        return pair.candidate.f1 - pair.baseline.f1
    return float(pair.candidate.tool_succeeded) - float(pair.baseline.tool_succeeded)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_bootstrap(
    pairs: Sequence[PredictionPair], *, seed: int, samples: int
) -> dict[str, object]:
    rng = random.Random(seed)
    metric_names: tuple[Metric, ...] = ("exact_match", "f1", "tool_success_rate")
    differences = {
        metric: [_metric_difference(pair, metric) for pair in pairs] for metric in metric_names
    }
    distributions: dict[str, list[float]] = {metric: [] for metric in differences}
    count = len(pairs)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        for metric, values in differences.items():
            distributions[metric].append(sum(values[index] for index in indices) / count)
    intervals = {
        metric: [
            _percentile(sorted(values), 0.025),
            _percentile(sorted(values), 0.975),
        ]
        for metric, values in distributions.items()
    }
    return {
        "method": "paired_percentile_bootstrap",
        "confidence": 0.95,
        "seed": seed,
        "samples": samples,
        "delta_95_ci": intervals,
    }


def _group_diagnostics(
    pairs: Sequence[PredictionPair],
    key: Callable[[PredictionPair], Sequence[str]],
) -> dict[str, object]:
    grouped: dict[str, list[PredictionPair]] = defaultdict(list)
    for pair in pairs:
        for value in key(pair):
            grouped[value].append(pair)
    output: dict[str, object] = {}
    for value, group in sorted(grouped.items()):
        baseline_metrics = _metrics(group, "baseline")
        candidate_metrics = _metrics(group, "candidate")
        output[value] = {
            "count": len(group),
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": {
                "exact_match": candidate_metrics["exact_match"] - baseline_metrics["exact_match"],
                "f1": candidate_metrics["f1"] - baseline_metrics["f1"],
                "tool_success_rate": candidate_metrics["tool_success_rate"]
                - baseline_metrics["tool_success_rate"],
            },
            "candidate_only_count": sum(
                pair.candidate.correct and not pair.baseline.correct for pair in group
            ),
            "baseline_only_count": sum(
                pair.baseline.correct and not pair.candidate.correct for pair in group
            ),
        }
    return output


def compare_predictions(
    baseline_path: Path,
    candidate_path: Path,
    *,
    output_path: Path | None = None,
    bootstrap_seed: int = 4242,
    bootstrap_samples: int = 10_000,
) -> dict[str, object]:
    """Compare two ordered KQAPro prediction files under one strict promotion gate."""

    baseline = _load_predictions(baseline_path)
    candidate = _load_predictions(candidate_path)
    pairs = _pair_predictions(baseline, candidate)
    if not pairs:
        raise ValueError("prediction files must contain at least one task")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")

    baseline_metrics = _metrics(pairs, "baseline")
    candidate_metrics = _metrics(pairs, "candidate")
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
    gates = {
        "candidate_correct_count_strictly_higher": (
            candidate_metrics["correct_count"] > baseline_metrics["correct_count"]
        ),
        "mean_f1_non_decreasing": candidate_metrics["f1"] >= baseline_metrics["f1"],
        "tool_success_rate_non_decreasing": (
            candidate_metrics["tool_success_rate"] >= baseline_metrics["tool_success_rate"]
        ),
        "baseline_model_request_failed_count_zero": (
            baseline_metrics["model_request_failed_count"] == 0
        ),
        "candidate_model_request_failed_count_zero": (
            candidate_metrics["model_request_failed_count"] == 0
        ),
    }
    transitions = Counter(
        f"{pair.baseline.diagnostic_reason}=>{pair.candidate.diagnostic_reason}" for pair in pairs
    )
    task_ids = [pair.baseline.task_id for pair in pairs]
    result: dict[str, object] = {
        "inputs": {
            "baseline": {
                "path": str(baseline_path),
                "source_sha256": file_hash(baseline_path),
            },
            "candidate": {
                "path": str(candidate_path),
                "source_sha256": file_hash(candidate_path),
            },
            "task_count": len(pairs),
            "task_order_sha256": stable_hash(task_ids),
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": {
            "correct_count": candidate_metrics["correct_count"] - baseline_metrics["correct_count"],
            "exact_match": candidate_metrics["exact_match"] - baseline_metrics["exact_match"],
            "f1": candidate_metrics["f1"] - baseline_metrics["f1"],
            "tool_success_rate": candidate_metrics["tool_success_rate"]
            - baseline_metrics["tool_success_rate"],
        },
        "evaluation_integrity": {
            "baseline_model_request_failed_count": baseline_metrics["model_request_failed_count"],
            "candidate_model_request_failed_count": candidate_metrics["model_request_failed_count"],
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
        "paired_bootstrap": _paired_bootstrap(
            pairs, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "diagnostics": {
            "by_baseline_reason": _group_diagnostics(
                pairs, lambda pair: (pair.baseline.diagnostic_reason,)
            ),
            "by_candidate_reason": _group_diagnostics(
                pairs, lambda pair: (pair.candidate.diagnostic_reason,)
            ),
            "by_operator_tag": _group_diagnostics(
                pairs,
                lambda pair: pair.baseline.operator_tags
                if pair.baseline.operator_tags
                else ("__none__",),
            ),
            "reason_transitions": dict(sorted(transitions.items())),
        },
        "gate": {
            "passed": all(gates.values()),
            "criteria": gates,
        },
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


__all__ = ["compare_predictions", "exact_mcnemar_p"]
