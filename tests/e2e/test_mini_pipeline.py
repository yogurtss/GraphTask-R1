from graphtask_r1.pipeline import run_mini_pipeline


def test_mini_pipeline_writes_replayable_artifacts(tmp_path) -> None:
    metrics = run_mini_pipeline(tmp_path, num_programs=30, seed=42)
    assert metrics["unrecoverable_errors"] == 0
    assert metrics["replay_accuracy"] == 1.0
    for name in (
        "programs.parquet",
        "tasks.parquet",
        "traces.parquet",
        "rejections.parquet",
        "metrics.json",
        "manifest.json",
    ):
        assert (tmp_path / name).exists()
