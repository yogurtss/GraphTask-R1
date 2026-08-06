from __future__ import annotations

from pathlib import Path
from typing import Any

from graphtask_r1.dsl import canonical_signature, compile_sparql, operator_tags, program_cost
from graphtask_r1.generation import ProgramSampler, compile_trace, verbalize
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import TaskCertificate, VerificationSummary
from graphtask_r1.utils import write_json, write_manifest, write_records
from graphtask_r1.verification import verify_task


def run_mini_pipeline(
    output_dir: Path,
    *,
    num_programs: int,
    seed: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = {
        "command": "e2e mini-pipeline",
        "graph": "toy",
        "num_programs": num_programs,
        "seed": seed,
        "dry_run": dry_run,
    }
    if dry_run:
        return config
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = toy_graph()
    sampled = ProgramSampler(backend, seed=seed).sample(num_programs)
    program_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    replay_correct = 0
    unrecoverable = 0
    for record in sampled:
        if record.program is None:
            rejection_rows.append({"index": record.index, "reason_code": record.reason_code})
            continue
        program = record.program
        signature = canonical_signature(program)
        program_rows.append(
            {
                "index": record.index,
                "program": program.model_dump(mode="json"),
                "program_signature": signature,
                "program_cost": program_cost(program),
            }
        )
        question = verbalize(program, backend)
        verification = verify_task(question, program, backend)
        if not verification.passed:
            for reason in verification.rejection_reasons:
                rejection_rows.append(
                    {"index": record.index, "program_signature": signature, "reason_code": reason}
                )
            continue
        try:
            answers = backend.execute_program(program)
            task_id = f"gt_toy_{seed}_{record.index:06d}"
            witnesses = backend.extract_witness(program, answers)
            task = TaskCertificate(
                task_id=task_id,
                source="toy_static",
                question=question,
                topic_entities=tuple(
                    backend.entity_info(entity_id)
                    for entity_id in sorted(_topic_ids(program.model_dump(mode="json")))
                ),
                program=program,
                sparql=compile_sparql(program),
                gold_answers=answers,
                witness_facts=tuple({fact for witness in witnesses for fact in witness.facts}),
                program_signature=signature,
                program_cost=program_cost(program),
                operator_tags=operator_tags(program),
                verification=VerificationSummary(
                    executable=True,
                    necessity_mean=verification.necessity_mean,
                    necessity_min=verification.necessity_min,
                    shortcut_found=verification.shortcut_found,
                    answer_leak=verification.answer_leak,
                ),
                generation={"seed": seed, "graph_snapshot": "toy-v1"},
            )
            trace = compile_trace(task_id, question, program, backend, seed=seed)
            replay_correct += int(trace.final_answers == answers)
            task_rows.append(task.model_dump(mode="json"))
            trace_rows.append(trace.model_dump(mode="json"))
        except (TypeError, ValueError, KeyError) as exc:
            unrecoverable += 1
            rejection_rows.append(
                {"index": record.index, "reason_code": "UNRECOVERABLE", "detail": str(exc)}
            )
    artifacts = [
        "programs.parquet",
        "tasks.parquet",
        "traces.parquet",
        "rejections.parquet",
        "metrics.json",
    ]
    write_records(output_dir / "programs.parquet", program_rows)
    write_records(output_dir / "tasks.parquet", task_rows)
    write_records(output_dir / "traces.parquet", trace_rows)
    write_records(output_dir / "rejections.parquet", rejection_rows)
    metrics = {
        "requested": num_programs,
        "sampled_valid": len(program_rows),
        "accepted": len(task_rows),
        "rejections": len(rejection_rows),
        "unrecoverable_errors": unrecoverable,
        "replay_accuracy": replay_correct / len(trace_rows) if trace_rows else 1.0,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_manifest(output_dir, config, artifacts)
    return metrics


def _topic_ids(program: dict[str, Any]) -> set[str]:
    if program["op"] == "entity":
        return {str(program["entity_id"])}
    if program["op"] == "all_entities":
        return set()
    if program["op"] in {"intersect", "union"}:
        return {entity for branch in program["inputs"] for entity in _topic_ids(branch)}
    return _topic_ids(program["input"])
