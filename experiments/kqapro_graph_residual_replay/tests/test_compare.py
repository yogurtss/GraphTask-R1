from __future__ import annotations

import json
from pathlib import Path

import pytest
from kqapro_grr.cli import main
from kqapro_grr.compare import compare_predictions, exact_mcnemar_p

from graphtask_r1.utils import read_json, write_records


def _prediction(
    task_id: str,
    *,
    exact_match: float,
    f1: float,
    tool_succeeded: bool,
    request_failed: bool = False,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "exact_match": exact_match,
        "f1": f1,
        "tool_succeeded": tool_succeeded,
        "rejection_reason": {"code": "MODEL_REQUEST_FAILED"} if request_failed else None,
    }


def _prediction_files(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.parquet"
    write_records(
        baseline,
        [
            _prediction("a", exact_match=1.0, f1=1.0, tool_succeeded=True),
            _prediction("b", exact_match=0.0, f1=0.0, tool_succeeded=False),
            _prediction("c", exact_match=0.0, f1=0.5, tool_succeeded=True),
        ],
    )
    write_records(
        candidate,
        [
            _prediction("c", exact_match=1.0, f1=1.0, tool_succeeded=True),
            _prediction("a", exact_match=1.0, f1=1.0, tool_succeeded=True),
            _prediction("b", exact_match=1.0, f1=1.0, tool_succeeded=True),
        ],
    )
    return baseline, candidate


def test_compare_pairs_by_task_id_and_passes_strict_gate(tmp_path: Path) -> None:
    baseline, candidate = _prediction_files(tmp_path)
    output = tmp_path / "comparison.json"

    result = compare_predictions(baseline, candidate, output_path=output)

    assert result["baseline"] == {
        "count": 3,
        "correct_count": 1,
        "exact_match": 1 / 3,
        "f1": 0.5,
        "tool_success_count": 2,
        "tool_success_rate": 2 / 3,
        "request_failure_count": 0,
    }
    assert result["candidate"] == {
        "count": 3,
        "correct_count": 3,
        "exact_match": 1.0,
        "f1": 1.0,
        "tool_success_count": 3,
        "tool_success_rate": 1.0,
        "request_failure_count": 0,
    }
    assert result["discordance"]["candidate_only_task_ids"] == ["b", "c"]
    assert result["discordance"]["baseline_only_task_ids"] == []
    assert result["discordance"]["exact_mcnemar_two_sided_p"] == 0.5
    assert result["gate"]["passed"] is True
    assert read_json(output) == result


@pytest.mark.parametrize(
    ("candidate_only", "baseline_only", "expected"),
    [(0, 0, 1.0), (2, 0, 0.5), (3, 0, 0.25), (2, 1, 1.0)],
)
def test_exact_mcnemar_p(candidate_only: int, baseline_only: int, expected: float) -> None:
    assert exact_mcnemar_p(candidate_only, baseline_only) == expected


def test_compare_rejects_different_task_sets(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.parquet"
    candidate = tmp_path / "candidate.parquet"
    write_records(
        baseline, [_prediction("a", exact_match=0.0, f1=0.0, tool_succeeded=False)]
    )
    write_records(
        candidate, [_prediction("b", exact_match=0.0, f1=0.0, tool_succeeded=False)]
    )

    with pytest.raises(ValueError, match="prediction task_id sets differ"):
        compare_predictions(baseline, candidate)


def test_request_failure_blocks_gate(tmp_path: Path) -> None:
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
                request_failed=True,
            )
        ],
    )
    write_records(
        candidate, [_prediction("a", exact_match=1.0, f1=1.0, tool_succeeded=True)]
    )

    result = compare_predictions(baseline, candidate)

    assert result["baseline"]["request_failure_count"] == 1
    assert result["gate"]["criteria"]["baseline_request_failures_zero"] is False
    assert result["gate"]["passed"] is False


def test_compare_cli_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline, candidate = _prediction_files(tmp_path)
    output = tmp_path / "comparison.json"

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
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["gate"]["passed"] is True
    assert read_json(output)["discordance"]["candidate_only_count"] == 2
