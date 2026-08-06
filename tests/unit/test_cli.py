import pytest

from graphtask_r1.cli import build_parser


def test_data_prepare_accepts_positive_worker_count() -> None:
    args = build_parser().parse_args(
        [
            "data",
            "prepare",
            "--dataset",
            "kqapro",
            "--raw-dir",
            "raw",
            "--output-dir",
            "processed",
            "--workers",
            "3",
        ]
    )
    assert args.workers == 3


def test_data_prepare_rejects_zero_workers() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "data",
                "prepare",
                "--dataset",
                "kqapro",
                "--raw-dir",
                "raw",
                "--output-dir",
                "processed",
                "--workers",
                "0",
            ]
        )
