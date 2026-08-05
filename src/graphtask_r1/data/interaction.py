from __future__ import annotations

from pathlib import Path

from graphtask_r1.graphscript import GraphScriptError, program_to_graphscript
from graphtask_r1.schema import TaskCertificate
from graphtask_r1.utils import write_json, write_records


def select_graphscript_tasks(
    tasks: list[TaskCertificate], output_path: Path
) -> dict[str, int]:
    selected: list[TaskCertificate] = []
    rejections: list[dict[str, object]] = []
    for task in tasks:
        reason: str | None = None
        if len(task.topic_entities) != 1:
            reason = "MULTIPLE_TOPICS"
        elif len(task.gold_answers.answers) != 1:
            reason = "NON_UNIQUE_RESULT"
        elif task.gold_answers.answers[0].kind != "entity":
            reason = "NON_ENTITY_ANSWER"
        elif task.verification.shortcut_found is True:
            reason = "SHORTCUT_FOUND"
        else:
            try:
                program_to_graphscript(task.program)
            except GraphScriptError as exc:
                reason = exc.reason_code
        if reason is None:
            selected.append(task)
        else:
            rejections.append({"task_id": task.task_id, "reason_code": reason})
    write_records(output_path, (task.model_dump(mode="json") for task in selected))
    write_records(output_path.with_name(output_path.stem + "_rejections.parquet"), rejections)
    metrics = {"input": len(tasks), "selected": len(selected), "rejected": len(rejections)}
    write_json(output_path.with_suffix(".metrics.json"), metrics)
    return metrics
