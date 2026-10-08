from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from graphtask_r1.schema import TaskCertificate
from graphtask_r1.utils import (
    file_hash,
    iter_record_json,
    stable_hash,
    write_json,
    write_records,
)


def _excluded_task_ids(roots: tuple[Path, ...]) -> set[str]:
    excluded: set[str] = set()
    for root in roots:
        for path in sorted(root.rglob("predictions.parquet")):
            table = pq.read_table(path, columns=["task_id"])
            excluded.update(str(value) for value in table["task_id"].to_pylist() if value)
    return excluded


def _stratum(task: TaskCertificate) -> tuple[str, str, str]:
    if task.program_cost <= 2:
        cost_bucket = "short"
    elif task.program_cost <= 5:
        cost_bucket = "medium"
    else:
        cost_bucket = "long"
    answer_kind = task.gold_answers.answers[0].kind if task.gold_answers.answers else "empty"
    return task.program.op, cost_bucket, answer_kind


def freeze_eval_sets(
    source_path: Path,
    output_dir: Path,
    *,
    exclude_roots: tuple[Path, ...],
    dev_count: int,
    blind_count: int,
    seed: int,
) -> dict[str, object]:
    manifest_path = output_dir / "eval_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen evaluation manifest: {manifest_path}")
    excluded = _excluded_task_ids(exclude_roots)
    buckets: dict[tuple[str, str, str], list[TaskCertificate]] = defaultdict(list)
    for raw in iter_record_json(source_path):
        task = TaskCertificate.model_validate_json(raw)
        if task.task_id not in excluded:
            buckets[_stratum(task)].append(task)
    ranked: list[tuple[float, str, TaskCertificate]] = []
    for stratum, values in sorted(buckets.items()):
        ordered = sorted(
            values,
            key=lambda task: stable_hash([seed, stratum, task.task_id]),
        )
        for index, task in enumerate(ordered):
            ranked.append(
                (
                    index / len(ordered),
                    stable_hash([seed, "eval", task.task_id]),
                    task,
                )
            )
    required = dev_count + blind_count
    selected = [task for _, _, task in sorted(ranked)[:required]]
    if len(selected) != required:
        raise ValueError(
            f"requested {required} unseen tasks, but only {len(selected)} remain after exclusion"
        )
    assignment_order = sorted(
        selected,
        key=lambda task: stable_hash([seed, "assignment", task.task_id]),
    )
    dev = assignment_order[:dev_count]
    blind = assignment_order[dev_count:]
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = output_dir / f"dev{dev_count}.parquet"
    blind_path = output_dir / f"blind{blind_count}.parquet"
    write_records(dev_path, (task.model_dump(mode="json") for task in dev))
    write_records(blind_path, (task.model_dump(mode="json") for task in blind))

    def counts(tasks: list[TaskCertificate]) -> dict[str, int]:
        values = Counter("|".join(_stratum(task)) for task in tasks)
        return dict(sorted(values.items()))

    manifest: dict[str, object] = {
        "source": str(source_path),
        "source_sha256": file_hash(source_path),
        "seed": seed,
        "excluded_task_count": len(excluded),
        "excluded_task_ids_sha256": stable_hash(sorted(excluded)),
        "excluded_prediction_roots": [str(root) for root in exclude_roots],
        "dev": {
            "path": str(dev_path),
            "count": len(dev),
            "task_ids": [task.task_id for task in dev],
            "strata": counts(dev),
        },
        "blind": {
            "path": str(blind_path),
            "count": len(blind),
            "task_ids": [task.task_id for task in blind],
            "strata": counts(blind),
        },
    }
    manifest["dev_sha256"] = file_hash(dev_path)
    manifest["blind_sha256"] = file_hash(blind_path)
    write_json(manifest_path, manifest)
    return manifest
