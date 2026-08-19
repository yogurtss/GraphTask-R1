from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from graphtask_r1.archive.store import TaskArchive


def promote_staged_tasks(
    staged_path: Path,
    archive_path: Path,
    *,
    min_pass_rate: float,
    max_pass_rate: float,
    min_novelty: float,
) -> dict[str, Any]:
    """Deterministically gate one round's candidates into the persistent archive."""
    if not 0.0 <= min_pass_rate <= max_pass_rate <= 1.0:
        raise ValueError("archive pass-rate bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= min_novelty <= 1.0:
        raise ValueError("archive min_novelty must be between 0 and 1")

    with TaskArchive(staged_path) as staged:
        candidates = sorted(
            staged.all(),
            key=lambda task: (task.program_signature, task.task_id),
        )

    decisions: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    accepted = 0
    with TaskArchive(archive_path) as archive:
        for task in candidates:
            pass_rate = float(task.solver_stats.get("pass_rate", 0.0))
            structural, textual = archive.novelty(task.program_signature, task.question)
            novelty = 0.5 * (structural + textual)
            reasons: list[str] = []
            if pass_rate < min_pass_rate:
                reasons.append("TOO_HARD")
            if pass_rate > max_pass_rate:
                reasons.append("TOO_EASY")
            if structural == 0.0:
                reasons.append("DUPLICATE_SIGNATURE")
            if novelty < min_novelty:
                reasons.append("LOW_NOVELTY")

            if not reasons:
                admission = {
                    "accepted": True,
                    "pass_rate": pass_rate,
                    "novelty_structural": structural,
                    "novelty_textual": textual,
                    "novelty": novelty,
                    "reason_codes": [],
                }
                promoted = task.model_copy(
                    update={
                        "solver_stats": {
                            **task.solver_stats,
                            "archive_admission": admission,
                        }
                    }
                )
                if archive.add(promoted):
                    accepted += 1
                else:  # Defensive: the single-writer path should make this unreachable.
                    reasons.append("DUPLICATE_SIGNATURE")

            if reasons:
                reason_counts.update(reasons)
            decisions.append(
                {
                    "task_id": task.task_id,
                    "program_signature": task.program_signature,
                    "accepted": not reasons,
                    "pass_rate": pass_rate,
                    "novelty_structural": structural,
                    "novelty_textual": textual,
                    "novelty": novelty,
                    "reason_codes": reasons,
                }
            )

    return {
        "candidates": len(candidates),
        "accepted": accepted,
        "rejected": len(candidates) - accepted,
        "reason_counts": dict(sorted(reason_counts.items())),
        "thresholds": {
            "min_pass_rate": min_pass_rate,
            "max_pass_rate": max_pass_rate,
            "min_novelty": min_novelty,
        },
        "decisions": decisions,
    }
