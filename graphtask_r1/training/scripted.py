from __future__ import annotations

from pathlib import Path
from typing import Any

from graphtask_r1.archive import TaskArchive
from graphtask_r1.generation import ProgramSampler, certify_proposal, compile_trace
from graphtask_r1.graph import toy_graph
from graphtask_r1.rewards import challenger_reward, solver_reward
from graphtask_r1.schema import TaskProposal
from graphtask_r1.utils import read_json, write_json, write_records


def _topic_ids(program: Any) -> tuple[str, ...]:
    value = program.model_dump(mode="json")

    def visit(node: dict[str, Any]) -> set[str]:
        if node["op"] == "entity":
            return {str(node["entity_id"])}
        if node["op"] == "all_entities":
            return set()
        if node["op"] in {"intersect", "union"}:
            return {entity for branch in node["inputs"] for entity in visit(branch)}
        return visit(node["input"])

    return tuple(sorted(visit(value)))


def run_scripted_selfplay(
    output_dir: Path,
    *,
    rounds: int = 3,
    candidates_per_round: int = 16,
    seed: int = 42,
    resume: bool = False,
) -> dict[str, Any]:
    """CPU contract test using real certificates/rewards, with no random reward stand-ins."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "archive.sqlite"
    manifest_path = output_dir / "manifest.json"
    start = 1
    if resume and manifest_path.exists():
        start = int(read_json(manifest_path)["last_completed_round"]) + 1
    backend = toy_graph()
    accepted_total = 0
    replayed_total = 0
    with TaskArchive(archive_path) as archive:
        for round_index in range(start, rounds + 1):
            records = ProgramSampler(backend, seed=seed + round_index).sample(candidates_per_round)
            questioner_rows: list[dict[str, Any]] = []
            solver_rows: list[dict[str, Any]] = []
            reward_rows: list[dict[str, Any]] = []
            for record in records:
                if record.program is None:
                    continue
                topics = _topic_ids(record.program)
                if not topics:
                    continue
                try:
                    task = certify_proposal(
                        TaskProposal(topic_entities=topics, program=record.program),
                        backend,
                        graph_snapshot="toy-v1",
                        round_index=round_index,
                    )
                except ValueError:
                    continue
                trace = compile_trace(
                    task.task_id, task.question, task.program, backend, seed=seed + record.index
                )
                replayed = trace.final_answers == task.gold_answers
                if not replayed:
                    raise RuntimeError(f"scripted trace replay failed: {task.task_id}")
                structural, textual = archive.novelty(task.program_signature, task.question)
                archive.add(task)
                verification = task.verification
                from graphtask_r1.schema import VerifierResult

                verifier = VerifierResult(
                    passed=True,
                    executable=True,
                    answer_nonempty=True,
                    cardinality_valid=True,
                    type_valid=True,
                    semantic_equivalent=True,
                    answer_leak=verification.answer_leak,
                    shortcut_found=verification.shortcut_found,
                    necessity_mean=verification.necessity_mean,
                    necessity_min=verification.necessity_min,
                    novelty_structural=structural,
                    novelty_textual=textual,
                )
                q_reward = challenger_reward(verifier, pass_rate=1.0, cost=task.program_cost)
                s_reward = solver_reward(
                    trace.final_answers,
                    task.gold_answers,
                    search_calls=max(0, len(trace.calls) - 1),
                    invalid_calls=0,
                )
                questioner_rows.append(
                    {"task_id": task.task_id, "role": "questioner", "round": round_index}
                )
                solver_rows.append(
                    {"task_id": task.task_id, "role": "solver", "round": round_index}
                )
                reward_rows.append(
                    {
                        "task_id": task.task_id,
                        "questioner": q_reward.model_dump(mode="json"),
                        "solver": s_reward.model_dump(mode="json"),
                    }
                )
                accepted_total += 1
                replayed_total += 1
            round_dir = output_dir / f"round_{round_index:03d}"
            write_records(round_dir / "questioner_rollouts.parquet", questioner_rows)
            write_records(round_dir / "solver_rollouts.parquet", solver_rows)
            write_records(round_dir / "reward_breakdown.parquet", reward_rows)
            write_json(
                round_dir / "manifest.json",
                {"round": round_index, "accepted": len(questioner_rows), "completed": True},
            )
            write_json(
                manifest_path,
                {
                    "last_completed_round": round_index,
                    "seed": seed,
                    "archive_size": len(archive.all()),
                },
            )
    return {
        "rounds_completed": rounds,
        "accepted": accepted_total,
        "replayed": replayed_total,
        "replay_accuracy": replayed_total / accepted_total if accepted_total else 1.0,
    }
