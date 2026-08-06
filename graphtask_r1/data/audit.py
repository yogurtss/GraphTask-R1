from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from graphtask_r1.schema import BenchmarkExample, TaskCertificate
from graphtask_r1.utils import read_records


def audit_records(path: Path, *, kind: str = "auto") -> dict[str, Any]:
    records = read_records(path)
    errors: list[dict[str, Any]] = []
    ids: list[str] = []
    splits: Counter[str] = Counter()
    accepted = 0
    for index, record in enumerate(records):
        try:
            selected = kind
            if selected == "auto":
                selected = "task" if "task_id" in record else "benchmark"
            if selected == "task":
                value = TaskCertificate.model_validate(record)
                ids.append(value.task_id)
                splits[value.split or "unknown"] += 1
                if not value.verification.executable or not value.gold_answers.answers:
                    raise ValueError("task is not executable or has no gold answers")
            elif selected == "benchmark":
                example = BenchmarkExample.model_validate(record)
                ids.append(example.example_id)
                splits[example.split] += 1
                if not example.question or not example.topic_entity_ids:
                    raise ValueError("benchmark example lacks question or gold topic entities")
            else:
                raise ValueError(f"unknown audit kind: {selected}")
            accepted += 1
        except (TypeError, ValueError, KeyError) as exc:
            errors.append({"index": index, "detail": str(exc)})
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    return {
        "path": str(path),
        "records": len(records),
        "valid": accepted,
        "invalid": len(errors),
        "duplicate_ids": duplicates,
        "splits": dict(splits),
        "errors": errors[:100],
        "passed": not errors and not duplicates,
    }
