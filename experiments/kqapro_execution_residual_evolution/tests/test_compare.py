from __future__ import annotations

import json
from pathlib import Path

import pytest
from kqapro_ere.cli import main
from kqapro_ere.compare import compare_predictions, exact_mcnemar_p

from graphtask_r1.utils import read_json, write_records


def _prediction(
    task_id: str,
    *,
    exact_match: float,
    f1: float,
    tool_succeeded: bool,
    reason: str | None,
    operator_tags: tuple[str, ...],
    rejection_code: str = "EVALUATION_FAILED",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "exact_match": exact_match,
        "f1": f1,
        "tool_succeeded": tool_succeeded,
        "rejection_reason": {"code": rejection_code, "cause_code": reason} if reason else None,
        "operator_tags": list(operator_tags),
    }


def _paired_files(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.parquet"
    write_records(
        baseline,
        [
            _prediction(
                "a",
                exact_match=1.0,
                f1=1.0,
                tool_succeeded=True,
                reason=None,
                operator_tags=("entity",),
            ),
            _prediction(
                "b",
                exact_match=0.0,
                f1=0.0,
                tool_succeeded=False,
                reason="EMPTY_RESULT",
                operator_tags=("entity", "hop"),
            ),
            _prediction(
                "c",
                exact_match=0.0,
                f1=0.5,
                tool_succeeded=True,
                reason="WRONG_DENOTATION",
                operator_tags=("count",),
            ),
            _prediction(
                "d",
                exact_match=0.0,
                f1=0.0,
                tool_succeeded=False,
                reason="ENTITY_NOT_FOUND",
                operator_tags=(),
            ),
        ],
    )
    write_records(
        candidate,
        [
            _prediction(
                "a",
                exact_match=1.0,
                f1=1.0,
                tool_succeeded=True,
                reason=None,
                operator_tags=("entity",),
            ),
            _prediction(
                "b",
                exact_match=1.0,
                f1=1.0,
                tool_succeeded=True,
                reason=None,
                operator_tags=("entity", "hop"),
            ),
            _prediction(
                "c",
                exact_match=1.0,
                f1=1.0,
                tool_succeeded=True,
                reason=None,
                operator_tags=("count",),
            ),
            _prediction(
                "d",
                exact_match=0.0,
                f1=0.0,
                tool_succeeded=False,
                reason="EMPTY_RESULT",
                operator_tags=(),
            ),
        ],
    )
    return baseline, candidate


def test_compare_reports_strict_gate_paired_tests_and_diagnostics(tmp_path: Path) -> None:
    baseline, candidate = _paired_files(tmp_path)
    output = tmp_path / "comparison.json"

    result = compare_predictions(
        baseline,
        candidate,
        output_path=output,
        bootstrap_seed=17,
        bootstrap_samples=200,
    )

    assert result["baseline"] == {
        "count": 4,
        "correct_count": 1,
        "exact_match": 0.25,
        "f1": 0.375,
        "tool_success_count": 2,
        "tool_success_rate": 0.5,
        "model_request_failed_count": 0,
    }
    assert result["candidate"] == {
        "count": 4,
        "correct_count": 3,
        "exact_match": 0.75,
        "f1": 0.75,
        "tool_success_count": 3,
        "tool_success_rate": 0.75,
        "model_request_failed_count": 0,
    }
    assert result["gate"]["passed"] is True
    assert result["evaluation_integrity"] == {
        "baseline_model_request_failed_count": 0,
        "candidate_model_request_failed_count": 0,
    }
    assert result["discordance"]["candidate_only_task_ids"] == ["b", "c"]
    assert result["discordance"]["baseline_only_task_ids"] == []
    assert result["discordance"]["exact_mcnemar_two_sided_p"] == 0.5
    diagnostics = result["diagnostics"]
    assert diagnostics["by_baseline_reason"]["EMPTY_RESULT"]["candidate_only_count"] == 1
    assert diagnostics["by_operator_tag"]["hop"]["candidate"]["exact_match"] == 1.0
    assert diagnostics["by_operator_tag"]["__none__"]["count"] == 1
    assert diagnostics["reason_transitions"]["EMPTY_RESULT=>EXACT_MATCH"] == 1
    inputs = result["inputs"]
    assert len(inputs["baseline"]["source_sha256"]) == 64
    assert len(inputs["candidate"]["source_sha256"]) == 64
    assert len(inputs["task_order_sha256"]) == 64
    assert read_json(output) == result

    repeated = compare_predictions(
        baseline,
        candidate,
        bootstrap_seed=17,
        bootstrap_samples=200,
    )
    assert repeated["paired_bootstrap"] == result["paired_bootstrap"]


@pytest.mark.parametrize(
    ("candidate_only", "baseline_only", "expected"),
    [(0, 0, 1.0), (2, 0, 0.5), (3, 0, 0.25), (2, 1, 1.0)],
)
def test_exact_mcnemar_p(candidate_only: int, baseline_only: int, expected: float) -> None:
    assert exact_mcnemar_p(candidate_only, baseline_only) == expected


def test_compare_rejects_different_task_order(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.parquet"
    records = [
        _prediction(
            task_id,
            exact_match=0.0,
            f1=0.0,
            tool_succeeded=False,
            reason="EMPTY_RESULT",
            operator_tags=("entity",),
        )
        for task_id in ("a", "b")
    ]
    write_records(baseline, records)
    write_records(candidate, list(reversed(records)))

    with pytest.raises(ValueError, match="ordered task_id mismatch at index 0"):
        compare_predictions(baseline, candidate, bootstrap_samples=10)


def test_compare_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.parquet"
    record = _prediction(
        "duplicate",
        exact_match=0.0,
        f1=0.0,
        tool_succeeded=False,
        reason="EMPTY_RESULT",
        operator_tags=("entity",),
    )
    write_records(baseline, [record, record])
    write_records(candidate, [record, record])

    with pytest.raises(ValueError, match="baseline predictions contain duplicate task_id"):
        compare_predictions(baseline, candidate, bootstrap_samples=10)


def test_model_request_failures_block_gate_and_are_counted(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.parquet"
    write_records(
        baseline,
        [
            _prediction(
                "a",
                exact_match=0.0,
                f1=0.0,
                tool_succeeded=False,
                reason="RuntimeError",
                operator_tags=("entity",),
                rejection_code="MODEL_REQUEST_FAILED",
            )
        ],
    )
    write_records(
        candidate,
        [
            _prediction(
                "a",
                exact_match=1.0,
                f1=1.0,
                tool_succeeded=True,
                reason=None,
                operator_tags=("entity",),
            )
        ],
    )

    result = compare_predictions(baseline, candidate, bootstrap_samples=10)

    assert result["evaluation_integrity"] == {
        "baseline_model_request_failed_count": 1,
        "candidate_model_request_failed_count": 0,
    }
    assert result["gate"]["criteria"]["baseline_model_request_failed_count_zero"] is False
    assert result["gate"]["criteria"]["candidate_model_request_failed_count_zero"] is True
    assert result["gate"]["passed"] is False


def test_compare_cli_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline, candidate = _paired_files(tmp_path)
    output = tmp_path / "cli-comparison.json"

    assert (
        main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--bootstrap-seed",
                "9",
                "--bootstrap-samples",
                "20",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["gate"]["passed"] is True
    assert read_json(output)["paired_bootstrap"]["seed"] == 9
