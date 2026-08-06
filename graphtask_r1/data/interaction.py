from __future__ import annotations

from pathlib import Path

from graphtask_r1.graph import GraphBackend
from graphtask_r1.graphscript import (
    GraphScriptError,
    execute_graphscript,
    graphscript_to_program,
    program_to_graphscript,
)
from graphtask_r1.schema import TaskCertificate
from graphtask_r1.training.relations import program_relations
from graphtask_r1.utils import write_json, write_records


def select_graphscript_tasks(
    tasks: list[TaskCertificate],
    output_path: Path,
    *,
    backend: GraphBackend,
    max_follow_limit: int = 100,
    max_edge_visits: int = 200,
    max_returned_entities: int = 1_000,
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
                script = program_to_graphscript(
                    task.program, follow_limit=max_follow_limit
                )
                compiled = graphscript_to_program(
                    script, seed_entity=task.topic_entities[0].entity_id
                )
                if compiled != task.program:
                    reason = "SEED_MISMATCH"
                full_answers = backend.execute_program(task.program)
                if reason is None and full_answers != task.gold_answers:
                    reason = "GOLD_MISMATCH"
                elif reason is None:
                    execution = execute_graphscript(
                        script,
                        backend,
                        seed_entity=task.topic_entities[0].entity_id,
                        allowed_relations=program_relations(task.program),
                        max_edge_visits=max_edge_visits,
                        max_returned_entities=max_returned_entities,
                        trace_id=f"select:{task.task_id}",
                    )
                    if execution.answers != full_answers:
                        reason = "BOUNDED_UNBOUNDED_MISMATCH"
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
