from pathlib import Path

from graphtask_r1.training.scripted import run_scripted_selfplay


def test_scripted_selfplay_three_rounds_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "selfplay"
    first = run_scripted_selfplay(output, rounds=3, candidates_per_round=12, seed=42)
    assert first["rounds_completed"] == 3
    assert first["accepted"] > 0
    assert first["replay_accuracy"] == 1.0
    resumed = run_scripted_selfplay(output, rounds=3, candidates_per_round=12, seed=42, resume=True)
    assert resumed["accepted"] == 0
    assert resumed["replay_accuracy"] == 1.0


def test_no_hardcoded_opponent_pass_rate() -> None:
    source_root = Path(__file__).parents[2] / "src"
    offenders = [
        path for path in source_root.rglob("*.py") if "opponent_pass_rate" in path.read_text()
    ]
    assert offenders == []
