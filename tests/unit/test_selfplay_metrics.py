from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from graphtask_r1.training.selfplay import _run_with_tee
from graphtask_r1.training.selfplay_metrics import (
    find_trainer_log,
    summarize_selfplay_round,
    write_selfplay_report,
)
from graphtask_r1.utils import read_json, write_json


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_run_with_tee_prints_and_persists_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "logs" / "train.log"

    _run_with_tee(
        [sys.executable, "-c", "print('visible self-play progress')"],
        env={},
        log_path=log_path,
    )

    captured = capsys.readouterr()
    assert "visible self-play progress" in captured.out
    assert log_path.read_text() == "visible self-play progress\n"


def test_selfplay_metrics_preserve_role_components_and_render_curves(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    round_dir = output_dir / "round_001"
    run_dir = round_dir / "v0"
    adapter = run_dir / "checkpoint-2"
    adapter.mkdir(parents=True)
    trainer_log = run_dir / "logging.jsonl"
    _write_jsonl(
        trainer_log,
        [
            {
                "loss": 0.4,
                "grad_norm": 1.2,
                "reward": 0.2,
                "reward_std": 0.3,
                "frac_reward_zero_std": 0.25,
                "kl": 0.01,
                "completions/min_length": 12.0,
                "completions/max_length": 40.0,
                "clip_ratio/high_max": 0.2,
                "entropy/mean": 1.5,
                "memory(GiB)": 7.5,
                "global_step/max_steps": "1/2",
            },
            {
                "loss": 0.2,
                "grad_norm": 0.8,
                "reward": 0.5,
                "completions/clipped_ratio": 0.1,
                "global_step/max_steps": "2/2",
            },
            {"eval_reward": 0.6, "eval_loss": 0.1, "global_step/max_steps": "2/2"},
            {
                "train_runtime": 10.0,
                "train_samples_per_second": 2.0,
                "train_loss": 0.3,
            },
        ],
    )
    stale = round_dir / "stale" / "logging.jsonl"
    _write_jsonl(stale, [{"loss": 99.0, "step": 99}])
    reward_dir = round_dir / "logs" / "metrics_attempt_001"
    _write_jsonl(
        reward_dir / "reward_components.rank-0.jsonl",
        [
            {
                "event": "graphtask_reward_components",
                "roles": {
                    "questioner": {
                        "samples": 2,
                        "means": {
                            "unweighted_score": 0.4,
                            "validity": 0.8,
                            "frontier": 0.7,
                            "novelty": 0.5,
                            "opponent_success_rate": 0.6,
                        },
                    },
                    "solver": {
                        "samples": 4,
                        "means": {
                            "unweighted_score": 0.75,
                            "f1": 0.75,
                            "exact_match": 0.5,
                        },
                    },
                },
            }
        ],
    )

    assert find_trainer_log(round_dir, adapter) == trainer_log
    summary = summarize_selfplay_round(
        1,
        {"questioner": 1, "solver": 1, "total": 2},
        trainer_log=trainer_log,
        reward_metrics_dir=reward_dir,
        archive_size_before=10,
        archive_size_after=13,
    )

    assert summary["cooperation_bottleneck"] == 0.4
    assert summary["roles"]["questioner"]["means"]["validity"] == 0.8
    assert summary["roles"]["solver"]["means"]["f1"] == 0.75
    assert summary["archive"] == {"size_before": 10, "size_after": 13, "added": 3}
    assert summary["trainer_metrics"]["loss"] == {
        "mean": 0.30000000000000004,
        "last": 0.2,
        "min": 0.2,
        "max": 0.4,
    }
    assert summary["training_history"][0]["frac_reward_zero_std"] == 0.25
    assert summary["training_history"][0]["completions/min_length"] == 12.0
    assert summary["training_history"][0]["clip_ratio/high_max"] == 0.2
    assert summary["training_history"][0]["entropy/mean"] == 1.5
    assert summary["training_history"][0]["memory(GiB)"] == 7.5
    assert "train_runtime" not in summary["trainer_metrics"]
    write_json(round_dir / "logs" / "metrics_summary.json", summary)

    artifacts = write_selfplay_report(output_dir)

    report = read_json(Path(artifacts["metrics"]))
    assert report["training_history"][-1]["global_step"] == 2
    assert Path(artifacts["round_metrics"]).read_text().count("\n") == 1
    assert Path(artifacts["plot"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_frontier_v2_combines_separate_phase_histories(tmp_path: Path) -> None:
    solver_log = tmp_path / "solver" / "logging.jsonl"
    questioner_log = tmp_path / "questioner" / "logging.jsonl"
    _write_jsonl(solver_log, [{"loss": 0.4, "reward": 0.3, "step": 1}])
    _write_jsonl(questioner_log, [{"loss": 0.2, "reward": 0.6, "step": 1}])

    summary = summarize_selfplay_round(
        1,
        {"questioner": 2, "solver": 2, "total": 4},
        trainer_log=None,
        trainer_logs={"solver": solver_log, "questioner": questioner_log},
        reward_metrics_dir=None,
    )

    assert [row["step"] for row in summary["training_history"]] == [1.0, 2.0]
    assert [row["phase_index"] for row in summary["training_history"]] == [0.0, 1.0]
    assert summary["trainer_logs"] == {
        "solver": str(solver_log),
        "questioner": str(questioner_log),
    }


def test_sparse_reward_components_use_all_role_samples_as_denominator(tmp_path: Path) -> None:
    reward_dir = tmp_path / "reward_metrics"
    _write_jsonl(
        reward_dir / "reward_components.rank-0.jsonl",
        [
            {
                "event": "graphtask_reward_components",
                "roles": {
                    "questioner": {
                        "samples": 4,
                        "means": {
                            "unweighted_score": 0.25,
                            "reject_non_json": 1.0,
                            "milestone_json_valid": 1.0,
                        },
                    }
                },
                "sample_components": [
                    {
                        "role": "questioner",
                        "reason_codes": ["NON_JSON"],
                        "components": {"reject_non_json": 1.0},
                    },
                    {
                        "role": "questioner",
                        "reason_codes": [],
                        "components": {"milestone_json_valid": 1.0},
                    },
                    {
                        "role": "questioner",
                        "reason_codes": [],
                        "components": {"milestone_json_valid": 1.0},
                    },
                    {
                        "role": "questioner",
                        "reason_codes": [],
                        "components": {"milestone_json_valid": 1.0},
                    },
                ],
            }
        ],
    )

    summary = summarize_selfplay_round(
        1,
        {"questioner": 1, "solver": 0, "total": 1},
        trainer_log=None,
        reward_metrics_dir=reward_dir,
    )

    means = summary["roles"]["questioner"]["means"]
    assert means["reject_non_json"] == 0.25
    assert means["milestone_json_valid"] == 0.75
    assert means["unweighted_score"] == 0.25
