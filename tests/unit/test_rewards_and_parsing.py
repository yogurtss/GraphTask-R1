import math
from pathlib import Path

import pytest

from graphtask_r1.archive import TaskArchive, promote_staged_tasks
from graphtask_r1.pipeline import run_mini_pipeline
from graphtask_r1.rewards import (
    challenger_reward,
    frontier_gated_challenger_reward,
    frontier_reward,
    normalize_advantages,
    questioner_rejection_reward,
)
from graphtask_r1.schema import TaskCertificate, VerifierResult
from graphtask_r1.training.parsing import parse_questioner_output, parse_solver_output
from graphtask_r1.utils import read_records


def test_frontier_peaks_at_target() -> None:
    assert frontier_reward(0.5) == 1.0
    assert frontier_reward(0.5) > frontier_reward(0.0)


def test_normalization_handles_zero_variance() -> None:
    assert normalize_advantages([1.0, 1.0]) == [0.0, 0.0]
    assert all(math.isfinite(value) for value in normalize_advantages([0.0, 1.0]))


def _verifier_result(*, passed: bool, reasons: tuple[str, ...] = ()) -> VerifierResult:
    return VerifierResult(
        passed=passed,
        executable=True,
        answer_nonempty=True,
        cardinality_valid=True,
        type_valid=True,
        semantic_equivalent=None,
        answer_leak="ANSWER_LEAK" in reasons,
        shortcut_found="SHORTCUT_FOUND" in reasons,
        necessity_mean=1.0,
        necessity_min=1.0,
        novelty_structural=1.0,
        novelty_textual=1.0,
        rejection_reasons=reasons,
    )


def test_questioner_rejection_stages_are_strictly_ordered() -> None:
    reasons = (
        "NON_JSON",
        "EXTRA_TEXT",
        "EXTRA_FIELD",
        "INVALID_HANDLE",
        "EMPTY_RESULT",
    )
    rewards = [questioner_rejection_reward(reason) for reason in reasons]

    assert [reward.total for reward in rewards] == [-1.0, -0.9, -0.75, -0.6, -0.4]
    assert [reward.components["reward_stage"] for reward in rewards] == [0, 1, 2, 3, 4]


def test_questioner_verification_and_quality_rewards_preserve_stage_order() -> None:
    leaked = challenger_reward(
        _verifier_result(passed=False, reasons=("ANSWER_LEAK",)),
        pass_rate=0.0,
        cost=1.0,
    )
    certified = challenger_reward(
        _verifier_result(passed=True),
        pass_rate=0.5,
        cost=0.0,
        target_alignment=1.0,
    )

    assert leaked.total == -0.3
    assert leaked.components["reward_stage"] == 5
    assert certified.total == 1.0
    assert certified.components["reward_stage"] == 6
    assert certified.total > leaked.total > questioner_rejection_reward("EMPTY_RESULT").total


def test_frontier_v2_is_opt_in_and_difficulty_dominates_quality() -> None:
    result = _verifier_result(passed=True)
    legacy_extreme = challenger_reward(
        result,
        pass_rate=0.0,
        cost=0.0,
        target_alignment=1.0,
    )
    v2_extreme = frontier_gated_challenger_reward(
        result,
        pass_rate=0.0,
        samples=8,
        cost=0.0,
        target_alignment=1.0,
    )
    v2_frontier = frontier_gated_challenger_reward(
        result,
        pass_rate=0.5,
        samples=8,
        cost=0.0,
        target_alignment=1.0,
    )

    assert legacy_extreme.total == pytest.approx(0.7705448641)
    assert v2_extreme.total < 0.2
    assert v2_frontier.total == 1.0
    assert v2_frontier.components["reward_variant_frontier_v2"] == 1.0
    assert v2_extreme.components["opponent_pass_rate_smoothed"] == 0.1


def test_frontier_v2_keeps_legacy_verification_penalties() -> None:
    failed = _verifier_result(passed=False, reasons=("ANSWER_LEAK",))
    reward = frontier_gated_challenger_reward(
        failed,
        pass_rate=0.0,
        samples=8,
        cost=1.0,
    )

    assert reward.total == -0.3
    assert reward.components["reward_stage"] == 5.0


def test_role_output_parsers() -> None:
    output = (
        '<task>{"question":"Who?","topic_entities":["alice"],'
        '"program":{"op":"entity","entity_id":"alice"}}</task>'
    )
    question, topics, program = parse_questioner_output(output)
    assert question == "Who?"
    assert topics == ("alice",)
    assert program.op == "entity"
    assert parse_solver_output('<answer>["alice"]</answer>').values() == ("alice",)
    literal = parse_solver_output('<answer>["Paris"]</answer>', answer_kind="literal")
    assert literal.answers[0].kind == "literal"


def test_archive_novelty_is_data_dependent(tmp_path: Path) -> None:
    run_mini_pipeline(tmp_path / "data", num_programs=20, seed=42)
    task = TaskCertificate.model_validate(read_records(tmp_path / "data" / "tasks.parquet")[0])
    with TaskArchive(tmp_path / "archive.sqlite") as archive:
        assert archive.novelty(task.program_signature, task.question) == (1.0, 1.0)
        assert archive.add(task)
        structural, textual = archive.novelty(task.program_signature, task.question)
        assert structural == 0.0
        assert textual == 0.0


def test_staged_archive_admission_is_deterministic_and_difficulty_gated(
    tmp_path: Path,
) -> None:
    run_mini_pipeline(tmp_path / "data", num_programs=20, seed=42)
    tasks = [
        TaskCertificate.model_validate(value)
        for value in read_records(tmp_path / "data" / "tasks.parquet")[:3]
    ]
    staged_path = tmp_path / "staged.sqlite"
    archive_path = tmp_path / "archive.sqlite"
    with TaskArchive(staged_path) as staged:
        for task, pass_rate in zip(tasks, (0.0, 0.5, 1.0), strict=True):
            assert staged.add(task.model_copy(update={"solver_stats": {"pass_rate": pass_rate}}))

    summary = promote_staged_tasks(
        staged_path,
        archive_path,
        min_pass_rate=0.25,
        max_pass_rate=0.75,
        min_novelty=0.0,
    )

    assert summary["candidates"] == 3
    assert summary["accepted"] == 1
    assert summary["reason_counts"] == {"TOO_EASY": 1, "TOO_HARD": 1}
    with TaskArchive(archive_path) as archive:
        promoted = archive.all()
    assert len(promoted) == 1
    assert promoted[0].solver_stats["archive_admission"]["accepted"] is True
