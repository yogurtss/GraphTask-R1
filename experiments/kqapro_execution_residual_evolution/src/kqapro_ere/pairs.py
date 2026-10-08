from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import TaskTrainingRecord
from graphtask_r1.training.prompts import role_prompt
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.utils import file_hash, iter_record_json, stable_hash, write_json
from kqapro_ere.mutation import (
    DEFAULT_SLOT_PRIORITY,
    CounterfactualMutator,
)


@dataclass(frozen=True)
class ResidualProfile:
    slot_priority: tuple[str, ...]
    observed_values: dict[str, tuple[object, ...]]
    effective_signal_count: int
    source: str | None = None


def load_residual_profile(path: Path | None) -> ResidualProfile:
    if path is None:
        return ResidualProfile(DEFAULT_SLOT_PRIORITY, {}, 0)
    raw = json.loads(path.read_text())
    priorities = tuple(str(value) for value in raw["slot_priority"])
    observed = {
        str(slot): tuple(values) for slot, values in dict(raw.get("observed_values", {})).items()
    }
    return ResidualProfile(
        priorities,
        observed,
        int(raw["effective_signal_count"]),
        str(path),
    )


def _rank_tasks(tasks: list[TaskTrainingRecord], *, seed: int) -> list[TaskTrainingRecord]:
    buckets: dict[tuple[str, str], list[TaskTrainingRecord]] = defaultdict(list)
    for task in tasks:
        answer_kind = task.gold_answers.answers[0].kind if task.gold_answers.answers else "empty"
        buckets[(task.program.op, answer_kind)].append(task)
    ranked: list[tuple[float, str, TaskTrainingRecord]] = []
    for stratum, values in sorted(buckets.items()):
        ordered = sorted(
            values,
            key=lambda task: stable_hash([seed, stratum, task.task_id]),
        )
        for index, task in enumerate(ordered):
            ranked.append(
                (
                    index / len(ordered),
                    stable_hash([seed, "interleave", task.task_id]),
                    task,
                )
            )
    return [task for _, _, task in sorted(ranked)]


def build_preference_pairs(
    tasks_path: Path,
    relation_catalog_path: Path,
    output_dir: Path,
    *,
    backend: GraphBackend,
    count: int,
    seed: int,
    residual_profile_path: Path | None,
    max_edge_visits: int,
    max_returned_entities: int,
) -> dict[str, object]:
    relations = load_relation_catalog(relation_catalog_path)
    profile = load_residual_profile(residual_profile_path)
    residual_guided = profile.effective_signal_count > 0
    generation_mode = (
        "execution_residual_guided_single_slot"
        if residual_guided
        else "dual_executed_default_single_slot_ablation"
    )
    slot_priority = profile.slot_priority if residual_guided else DEFAULT_SLOT_PRIORITY
    observed_values = profile.observed_values if residual_guided else {}
    tasks = [TaskTrainingRecord.model_validate_json(raw) for raw in iter_record_json(tasks_path)]
    mutator = CounterfactualMutator(
        backend,
        relations,
        seed=seed,
        max_edge_visits=max_edge_visits,
        max_returned_entities=max_returned_entities,
        observed_values=observed_values,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    certificate_path = output_dir / "counterfactual_certificates.jsonl"
    attempt_rejections_path = output_dir / "mutation_attempt_rejections.jsonl"
    reason_counts: Counter[str] = Counter()
    attempt_reason_counts: Counter[str] = Counter()
    attempt_stage_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    selected_task_ids: list[str] = []
    with (
        train_path.open("w", encoding="utf-8") as train_stream,
        certificate_path.open("w", encoding="utf-8") as certificate_stream,
        attempt_rejections_path.open("w", encoding="utf-8") as attempt_stream,
    ):
        for task in _rank_tasks(tasks, seed=seed):
            outcome = mutator.mutate(task, slot_priority=slot_priority)
            reason_counts[outcome.reason] += 1
            for attempt in outcome.attempt_rejections:
                attempt_stream.write(
                    json.dumps(attempt.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                )
                attempt_reason_counts[attempt.reason_code] += 1
                attempt_stage_counts[attempt.stage] += 1
            if outcome.certificate is None:
                continue
            certificate = outcome.certificate
            messages = role_prompt(
                "solver",
                f"Question: {task.question}",
                interaction_mode="graphscript",
                relation_catalog=relations,
                graphscript_version="0.3",
            )
            messages.append({"role": "assistant", "content": certificate.chosen_graphscript})
            pair: dict[str, Any] = {
                "messages": messages,
                "rejected_response": certificate.rejected_graphscript,
                "task_id": task.task_id,
                "ere_edit_slot": certificate.edit.slot,
            }
            train_stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
            certificate_stream.write(
                json.dumps(certificate.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            selected_task_ids.append(task.task_id)
            slot_counts[certificate.edit.slot] += 1
            if len(selected_task_ids) >= count:
                break
    metrics: dict[str, object] = {
        "algorithm": "execution_residual_evolution_v1",
        "generation_mode": generation_mode,
        "residual_guided": residual_guided,
        "residual_effective_signal_count": profile.effective_signal_count,
        "source_tasks": str(tasks_path),
        "source_tasks_sha256": file_hash(tasks_path),
        "relation_catalog": str(relation_catalog_path),
        "relation_catalog_sha256": file_hash(relation_catalog_path),
        "residual_profile": profile.source,
        "residual_profile_sha256": (
            file_hash(residual_profile_path) if residual_profile_path is not None else None
        ),
        "seed": seed,
        "requested_pairs": count,
        "selected_pairs": len(selected_task_ids),
        "selected_task_ids": selected_task_ids,
        "slot_counts": dict(sorted(slot_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "attempt_rejection_count": sum(attempt_reason_counts.values()),
        "attempt_reason_counts": dict(sorted(attempt_reason_counts.items())),
        "attempt_stage_counts": dict(sorted(attempt_stage_counts.items())),
        "max_edge_visits": max_edge_visits,
        "max_returned_entities": max_returned_entities,
        "train": str(train_path),
        "train_sha256": file_hash(train_path),
        "certificates": str(certificate_path),
        "certificates_sha256": file_hash(certificate_path),
        "mutation_attempt_rejections": str(attempt_rejections_path),
        "mutation_attempt_rejections_sha256": file_hash(attempt_rejections_path),
    }
    write_json(output_dir / "manifest.json", metrics)
    return metrics
