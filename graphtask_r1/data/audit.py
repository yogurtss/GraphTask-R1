from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from graphtask_r1.schema import BenchmarkExample, TaskCertificate, TaskTrainingRecord
from graphtask_r1.utils import ProgressLogger, RecordWriter, iter_record_json, record_count

GRAPH_QA_DATASETS = frozenset({"webqsp", "cwq", "grailqa"})


def audit_records(
    path: Path,
    *,
    kind: str = "auto",
    deep: bool = False,
    training_view_output: Path | None = None,
) -> dict[str, Any]:
    if training_view_output is not None:
        if kind != "task":
            raise ValueError("a training view requires --kind task")
        if path.resolve() == training_view_output.resolve():
            raise ValueError("training view output must differ from the audit input")
    total = record_count(path)
    errors: list[dict[str, Any]] = []
    ids: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    accepted = 0
    progress = ProgressLogger("data.audit.records", total=total)
    progress.start(
        path=str(path),
        kind=kind,
        deep=deep,
        loading="streaming",
        training_view_output=(
            str(training_view_output) if training_view_output is not None else None
        ),
    )
    writer_context = (
        RecordWriter(training_view_output)
        if training_view_output is not None
        else nullcontext(None)
    )
    with writer_context as writer:
        for index, raw in enumerate(iter_record_json(path)):
            try:
                selected = kind
                task_value: TaskCertificate | TaskTrainingRecord | None = None
                if selected == "auto":
                    try:
                        task_value = TaskTrainingRecord.model_validate_json(raw)
                        selected = "task"
                    except ValueError:
                        selected = "benchmark"
                if selected == "task":
                    model = TaskCertificate if deep else TaskTrainingRecord
                    value = (
                        task_value
                        if task_value is not None and not deep
                        else model.model_validate_json(raw)
                    )
                    ids[value.task_id] += 1
                    splits[value.split or "unknown"] += 1
                    if not value.verification.executable or not value.gold_answers.answers:
                        raise ValueError("task is not executable or has no gold answers")
                    if writer is not None:
                        training_value = (
                            value
                            if isinstance(value, TaskTrainingRecord)
                            else TaskTrainingRecord.model_validate(value.model_dump())
                        )
                        writer.write(training_value.model_dump(mode="json"))
                elif selected == "benchmark":
                    if writer is not None:
                        raise ValueError("cannot write a training view from benchmark records")
                    example = BenchmarkExample.model_validate_json(raw)
                    ids[example.example_id] += 1
                    splits[example.split] += 1
                    if example.dataset in GRAPH_QA_DATASETS and not example.topic_entity_ids:
                        raise ValueError("graph benchmark example lacks gold topic entities")
                    if example.dataset not in GRAPH_QA_DATASETS and (
                        not example.gold_answers.answers or not example.answer_aliases
                    ):
                        raise ValueError("open-QA benchmark example lacks gold answer aliases")
                else:
                    raise ValueError(f"unknown audit kind: {selected}")
                accepted += 1
            except (TypeError, ValueError, KeyError) as exc:
                errors.append({"index": index, "detail": str(exc)})
            progress.update(index + 1, valid=accepted, invalid=len(errors))
    progress.finish(total, valid=accepted, invalid=len(errors))
    duplicates = sorted(value for value, count in ids.items() if count > 1)
    return {
        "path": str(path),
        "records": total,
        "valid": accepted,
        "invalid": len(errors),
        "deep": deep,
        "training_view_output": (
            str(training_view_output) if training_view_output is not None else None
        ),
        "duplicate_ids": duplicates,
        "splits": dict(splits),
        "errors": errors[:100],
        "passed": not errors and not duplicates,
    }
