import math
from pathlib import Path

from graphtask_r1.archive import TaskArchive
from graphtask_r1.pipeline import run_mini_pipeline
from graphtask_r1.rewards import (
    challenger_reward,
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


def test_archive_novelty_is_data_dependent(tmp_path: Path) -> None:
    run_mini_pipeline(tmp_path / "data", num_programs=20, seed=42)
    task = TaskCertificate.model_validate(read_records(tmp_path / "data" / "tasks.parquet")[0])
    with TaskArchive(tmp_path / "archive.sqlite") as archive:
        assert archive.novelty(task.program_signature, task.question) == (1.0, 1.0)
        assert archive.add(task)
        structural, textual = archive.novelty(task.program_signature, task.question)
        assert structural == 0.0
        assert textual == 0.0
