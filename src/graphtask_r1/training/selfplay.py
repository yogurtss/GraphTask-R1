from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from graphtask_r1.utils import read_json, write_json, write_manifest, write_records


def run_orchestration_smoke(
    output_dir: Path,
    *,
    rounds: int,
    questioner_groups: int,
    solver_episodes: int,
    seed: int,
    model: str,
    resume: bool,
) -> dict[str, Any]:
    """Exercise round/checkpoint contracts without pretending to train a language model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_round = 1
    state = {"shared_parameter_version": 0, "archive_size": 0}
    if resume and (output_dir / "manifest.json").exists():
        manifest = read_json(output_dir / "manifest.json")
        start_round = int(manifest["last_completed_round"]) + 1
        state = dict(manifest["state"])
    rng = random.Random(seed + start_round)
    for round_index in range(start_round, rounds + 1):
        round_dir = output_dir / f"round_{round_index}"
        q_rewards = [rng.random() for _ in range(questioner_groups)]
        s_rewards = [rng.random() for _ in range(solver_episodes)]
        q_rows = [
            {"round": round_index, "role": "questioner", "index": i, "reward": reward}
            for i, reward in enumerate(q_rewards)
        ]
        s_rows = [
            {"round": round_index, "role": "solver", "index": i, "reward": reward}
            for i, reward in enumerate(s_rewards)
        ]
        write_records(round_dir / "questioner_rollouts.parquet", q_rows)
        write_records(round_dir / "solver_rollouts.parquet", s_rows)
        write_records(round_dir / "task_archive.parquet", q_rows)
        q_mean = sum(q_rewards) / max(1, len(q_rewards))
        s_mean = sum(s_rewards) / max(1, len(s_rewards))
        write_json(
            round_dir / "reward_breakdown.json",
            {"questioner": {"total": q_mean}, "solver": {"total": s_mean}},
        )
        write_json(
            round_dir / "gradient_diagnostics.json",
            {
                "questioner_norm": 0.0,
                "solver_norm": 0.0,
                "cosine_similarity": 0.0,
                "note": "orchestration smoke only; real gradients are emitted by verl",
            },
        )
        checkpoint = output_dir / f"checkpoints/shared_policy_round_{round_index}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        state["shared_parameter_version"] = round_index
        state["archive_size"] = int(state["archive_size"]) + len(q_rows)
        write_json(
            checkpoint / "adapter_manifest.json",
            {"model": model, "round": round_index, "shared": True},
        )
        write_json(
            round_dir / "manifest.json",
            {"round": round_index, "seed": seed, "state": state, "resumable": True},
        )
        write_json(
            output_dir / "manifest.json",
            {"last_completed_round": round_index, "seed": seed, "state": state, "resumable": True},
        )
    write_manifest(
        output_dir,
        {
            "command": "train mini-self-play",
            "rounds": rounds,
            "questioner_groups": questioner_groups,
            "solver_episodes": solver_episodes,
            "seed": seed,
            "model": model,
            "backend": "orchestration-smoke",
        },
        ["round_*/", "checkpoints/"],
    )
    return {"rounds_completed": rounds, **state}
