from pathlib import Path

import pytest

from graphtask_r1.cli import _launch_stage, build_parser


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


def test_sample_seeds_defaults_to_kqapro_snapshot() -> None:
    args = build_parser().parse_args(
        ["data", "sample-seeds", "--output", "seeds.parquet"]
    )
    assert args.snapshot == "kqapro-v1"
    assert args.count == 256


def test_cuda124_training_profile_is_forwarded_to_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NUM_GPUS", "1")
    config = tmp_path / "sft.yaml"
    config.write_text(
        "\n".join(
            [
                "verl_profile: cuda124",
                "model_path: model",
                "train_data: train.parquet",
                "val_data: val.parquet",
            ]
        )
        + "\n"
    )
    result = _launch_stage("sft", config, dry_run=True)
    assert result["command"] == ["bash", "scripts/train_sft.sh"]
    assert result["environment"]["VERL_PROFILE"] == "cuda124"
    assert result["environment"]["NUM_GPUS"] == "1"


def test_ms_swift_profile_selects_ms_swift_launcher(tmp_path: Path) -> None:
    config = tmp_path / "sft.yaml"
    config.write_text(
        "\n".join(
            [
                "training_backend: ms_swift",
                "model_path: model",
                "model_type: qwen3",
                "train_data: train.parquet",
                "val_data: val.parquet",
            ]
        )
        + "\n"
    )

    result = _launch_stage("sft", config, dry_run=True)

    assert result["training_backend"] == "ms_swift"
    assert result["command"] == ["bash", "scripts/train_ms_swift_sft.sh"]
    assert result["environment"]["MODEL_TYPE"] == "qwen3"


def test_unknown_training_backend_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "sft.yaml"
    config.write_text("training_backend: unknown\n")
    with pytest.raises(ValueError, match="unsupported training backend"):
        _launch_stage("sft", config, dry_run=True)
