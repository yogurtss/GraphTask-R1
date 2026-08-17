from pathlib import Path

import pytest

from graphtask_r1.cli import _launch_stage, build_parser, main
from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import Entity, Hop, RelationInfo, TaskProposal
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.utils import write_json, write_records


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
    assert args.max_witness_facts == 0
    assert args.train_sample_size == 20_000
    assert args.trace_mode == "none"
    assert args.verification_mode == "source"


def test_data_audit_accepts_deep_and_training_view_output() -> None:
    args = build_parser().parse_args(
        [
            "data",
            "audit",
            "--input",
            "tasks.parquet",
            "--kind",
            "task",
            "--deep",
            "--training-view-output",
            "training_tasks.parquet",
        ]
    )
    assert args.deep is True
    assert args.training_view_output == Path("training_tasks.parquet")


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


def test_data_prepare_accepts_ssp_bucket_filter_and_kilt_text_flag() -> None:
    ssp = build_parser().parse_args(
        [
            "data",
            "prepare",
            "--dataset",
            "ssp",
            "--raw-dir",
            "raw",
            "--output-dir",
            "processed",
            "--include-datasets",
            "hotpotqa,triviaqa",
        ]
    )
    assert ssp.include_datasets == "hotpotqa,triviaqa"

    kilt = build_parser().parse_args(
        [
            "data",
            "prepare",
            "--dataset",
            "kilt",
            "--raw-dir",
            "kilt.json",
            "--output-dir",
            "processed",
            "--no-text-index",
        ]
    )
    assert kilt.no_text_index is True


def test_kilt_grpo_bootstrap_cli_has_bounded_reproducible_defaults() -> None:
    args = build_parser().parse_args(
        [
            "data",
            "bootstrap-kilt-grpo",
            "--output-dir",
            "processed",
            "--count",
            "16",
            "--seed",
            "7",
            "--dry-run",
        ]
    )
    assert args.snapshot == "kilt-2019-08-01-v1"
    assert args.count == 16
    assert args.seed == 7
    assert args.pool_limit == 100_000
    assert args.val_ratio == 0.1


def test_sft_export_accepts_graphscript_v02() -> None:
    args = build_parser().parse_args(
        [
            "data",
            "export-sft",
            "--input",
            "tasks.parquet",
            "--output",
            "sft.parquet",
            "--interaction-mode",
            "graphscript",
            "--graphscript-version",
            "0.2",
        ]
    )
    assert args.graphscript_version == "0.2"


def test_questioner_sft_export_and_balanced_combine_options() -> None:
    args = build_parser().parse_args(
        [
            "data",
            "export-questioner-sft",
            "--input",
            "tasks.parquet",
            "--output",
            "questioner.parquet",
            "--count",
            "2048",
            "--seed",
            "7",
        ]
    )
    assert args.count == 2048
    assert args.seed == 7
    assert args.graphscript_version == "0.3"
    assert args.interaction_mode == "graphscript"

    combined = build_parser().parse_args(
        [
            "data",
            "combine-sft",
            "--solver-input",
            "solver.parquet",
            "--questioner-input",
            "questioner.parquet",
            "--output",
            "mixed.parquet",
            "--solver-weight",
            "1",
            "--questioner-weight",
            "1",
        ]
    )
    assert combined.solver_input == Path("solver.parquet")
    assert combined.questioner_input == Path("questioner.parquet")
    assert combined.solver_weight == 1
    assert combined.questioner_weight == 1

    seeds = build_parser().parse_args(
        [
            "data",
            "export-questioner-seeds",
            "--input",
            "tasks.parquet",
            "--output",
            "seeds.parquet",
            "--count",
            "4096",
            "--relation-catalog",
            "relations.json",
        ]
    )
    assert seeds.count == 4096
    assert seeds.max_topic_entities is None
    assert seeds.graphscript_version == "0.3"


def test_questioner_sft_cli_exports_requested_rows(tmp_path: Path) -> None:
    task = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
    )
    tasks = tmp_path / "tasks.parquet"
    catalog = tmp_path / "relations.json"
    output = tmp_path / "questioner.parquet"
    write_records(tasks, [task.model_dump(mode="json")])
    write_json(
        catalog,
        [RelationInfo(relation_id="works_at", label="works at").model_dump(mode="json")],
    )

    assert (
        main(
            [
                "data",
                "export-questioner-sft",
                "--input",
                str(tasks),
                "--output",
                str(output),
                "--count",
                "1",
                "--relation-catalog",
                str(catalog),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert output.with_suffix(".metrics.json").is_file()


def test_relation_catalog_unions_multiple_task_inputs(tmp_path: Path) -> None:
    train_task = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
    )
    val_task = certify_proposal(
        TaskProposal(
            topic_entities=("acme",),
            program=Hop(input=Entity(entity_id="acme"), relation="located_in"),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
    )
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    output = tmp_path / "relations.json"
    write_records(train_path, [train_task.model_dump(mode="json")])
    write_records(val_path, [val_task.model_dump(mode="json")])

    assert (
        main(
            [
                "data",
                "build-relation-catalog",
                "--input",
                str(train_path),
                str(val_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert [relation.relation_id for relation in load_relation_catalog(output)] == [
        "age",
        "country",
        "friend",
        "friend_of_friend",
        "located_in",
        "works_at",
    ]


def test_kilt_grpo_profile_starts_from_separate_kilt_sft_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KILT_SFT_ADAPTER", "/models/kilt-sft")
    monkeypatch.setenv("KILT_GRPO_TRAIN_DATA", "/data/kilt/train.parquet")
    monkeypatch.setenv("KILT_GRPO_VAL_DATA", "/data/kilt/val.parquet")
    monkeypatch.setenv("KILT_GRPO_OUTPUT_DIR", "/outputs/kilt-grpo")

    result = _launch_stage(
        "solver-grpo",
        Path("configs/experiments/qwen3_4b_kilt_solver_grpo_ms_swift_cuda124.yaml"),
        dry_run=True,
    )

    assert result["training_backend"] == "ms_swift"
    assert result["environment"]["LORA_ADAPTER_PATH"] == "/models/kilt-sft"
    assert result["environment"]["TRAIN_DATA"] == "/data/kilt/train.parquet"
    assert result["environment"]["VAL_DATA"] == "/data/kilt/val.parquet"


def test_sample_seeds_defaults_to_kqapro_snapshot() -> None:
    args = build_parser().parse_args(["data", "sample-seeds", "--output", "seeds.parquet"])
    assert args.snapshot == "kqapro-v1"
    assert args.count == 256
    assert args.graphscript_version == "0.3"
    assert args.max_seed_neighbor_facts == 200
    assert args.max_seed_relations == 64


def test_kqapro_eval_and_visualization_cli_are_separate() -> None:
    evaluation = build_parser().parse_args(
        [
            "evaluate",
            "kqapro-val",
            "--config",
            "configs/evaluation/kqapro_val.yaml",
            "--model-stage",
            "base_tool",
        ]
    )
    assert evaluation.output_dir is None
    assert evaluation.limit is None
    assert evaluation.model_stage == "base_tool"

    visualization = build_parser().parse_args(
        [
            "visualize",
            "kqapro",
            "--config",
            "configs/evaluation/kqapro_val.yaml",
            "--model-stage",
            "grpo",
            "--indices",
            "0,12,41",
            "--inspect-only",
        ]
    )
    assert visualization.output_dir is None
    assert visualization.indices == "0,12,41"
    assert visualization.model_stage == "grpo"
    assert visualization.limit == 3
    assert visualization.inspect_only is True

    comparison = build_parser().parse_args(
        [
            "evaluate",
            "kqapro-compare",
            "--metrics",
            "base.json",
            "base_tool.json",
            "--baseline-stage",
            "base_tool",
        ]
    )
    assert comparison.metrics == [Path("base.json"), Path("base_tool.json")]
    assert comparison.baseline_stage == "base_tool"


def test_training_launcher_defaults_to_ms_swift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NUM_GPUS", "1")
    config = tmp_path / "sft.yaml"
    config.write_text(
        "\n".join(
            [
                "model_path: model",
                "model_type: qwen3",
                "train_data: train.parquet",
                "val_data: val.parquet",
                "micro_batch_size: 2",
                "eval_batch_size: 3",
                "gradient_accumulation_steps: 4",
            ]
        )
        + "\n"
    )
    result = _launch_stage("sft", config, dry_run=True)
    assert result["training_backend"] == "ms_swift"
    assert result["command"] == ["bash", "scripts/train_ms_swift_sft.sh"]
    assert result["environment"]["MODEL_TYPE"] == "qwen3"
    assert result["environment"]["NUM_GPUS"] == "1"
    assert result["environment"]["MICRO_BATCH_SIZE"] == "2"
    assert result["environment"]["EVAL_BATCH_SIZE"] == "3"
    assert result["environment"]["GRADIENT_ACCUMULATION_STEPS"] == "4"


def test_grpo_batch_config_and_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "grpo.yaml"
    config.write_text(
        "\n".join(
            [
                "training_backend: ms_swift",
                "model_path: model",
                "model_type: qwen3",
                "lora_adapter_path: adapter",
                "train_data: train.parquet",
                "val_data: val.parquet",
                "micro_batch_size: 1",
                "eval_batch_size: 4",
                "gradient_accumulation_steps: 4",
                "steps_per_generation: 8",
                "rollout_n: 4",
            ]
        )
        + "\n"
    )
    monkeypatch.setenv("MICRO_BATCH_SIZE", "3")

    result = _launch_stage("solver-grpo", config, dry_run=True)

    assert result["environment"]["MICRO_BATCH_SIZE"] == "3"
    assert result["environment"]["EVAL_BATCH_SIZE"] == "4"
    assert result["environment"]["GRADIENT_ACCUMULATION_STEPS"] == "4"
    assert result["environment"]["STEPS_PER_GENERATION"] == "8"
    assert result["environment"]["ROLLOUT_N"] == "4"


def test_training_batch_config_rejects_non_positive_values(tmp_path: Path) -> None:
    config = tmp_path / "sft.yaml"
    config.write_text("training_backend: ms_swift\nmicro_batch_size: 0\n")

    with pytest.raises(ValueError, match="MICRO_BATCH_SIZE must be a positive integer"):
        _launch_stage("sft", config, dry_run=True)


def test_grpo_batch_config_rejects_incompatible_generation_groups(tmp_path: Path) -> None:
    config = tmp_path / "grpo.yaml"
    config.write_text(
        "\n".join(
            [
                "training_backend: ms_swift",
                "num_gpus: 3",
                "micro_batch_size: 1",
                "eval_batch_size: 1",
                "gradient_accumulation_steps: 4",
                "steps_per_generation: 4",
                "rollout_n: 4",
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="EVAL_BATCH_SIZE must be divisible"):
        _launch_stage("solver-grpo", config, dry_run=True)


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


def test_non_ms_swift_training_backend_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "sft.yaml"
    config.write_text("training_backend: unknown\n")
    with pytest.raises(ValueError, match="only ms_swift is supported"):
        _launch_stage("sft", config, dry_run=True)


def test_qwen3_8b_example_configs_are_dry_run_only_and_fully_mapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapped_variables = (
        "MODEL_PATH",
        "NUM_GPUS",
        "MICRO_BATCH_SIZE",
        "EVAL_BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "STEPS_PER_GENERATION",
        "MAX_LENGTH",
        "MAX_COMPLETION_LENGTH",
        "LORA_RANK",
        "LORA_ALPHA",
        "LR",
        "EPOCHS",
        "VLLM_MODE",
        "ROLLOUT_N",
        "TRAIN_CUDA_VISIBLE_DEVICES",
    )
    for variable in mapped_variables:
        monkeypatch.delenv(variable, raising=False)

    sft = _launch_stage(
        "sft",
        Path("configs/experiments/qwen3_8b_sft_ms_swift_cuda124.yaml"),
        dry_run=True,
    )
    grpo = _launch_stage(
        "solver-grpo",
        Path("configs/experiments/qwen3_8b_solver_grpo_ms_swift_cuda124.yaml"),
        dry_run=True,
    )

    assert sft["command"] == ["bash", "scripts/train_ms_swift_sft.sh"]
    assert sft["environment"]["MODEL_PATH"] == "Qwen/Qwen3-8B"
    assert sft["environment"]["NUM_GPUS"] == "4"
    assert sft["environment"]["MICRO_BATCH_SIZE"] == "2"
    assert sft["environment"]["GRADIENT_ACCUMULATION_STEPS"] == "8"
    assert sft["environment"]["MAX_LENGTH"] == "32768"
    assert sft["environment"]["LR"] == "0.00002"
    assert sft["environment"]["LORA_RANK"] == "32"
    assert grpo["command"] == ["bash", "scripts/train_ms_swift_grpo.sh"]
    assert grpo["environment"]["MODEL_PATH"] == "Qwen/Qwen3-8B"
    assert grpo["environment"]["NUM_GPUS"] == "3"
    assert grpo["environment"]["MICRO_BATCH_SIZE"] == "2"
    assert grpo["environment"]["EVAL_BATCH_SIZE"] == "4"
    assert grpo["environment"]["GRADIENT_ACCUMULATION_STEPS"] == "4"
    assert grpo["environment"]["STEPS_PER_GENERATION"] == "4"
    assert grpo["environment"]["MAX_COMPLETION_LENGTH"] == "4096"
    assert grpo["environment"]["LR"] == "0.000002"
    assert grpo["environment"]["ROLLOUT_N"] == "4"
    assert grpo["environment"]["VLLM_MODE"] == "server"
    assert grpo["environment"]["TRAIN_CUDA_VISIBLE_DEVICES"] == "1,2,3"


@pytest.mark.parametrize("stage", ["sft", "solver-grpo"])
def test_learning_rate_scales_linearly_with_micro_batch(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LR", raising=False)
    monkeypatch.delenv("MICRO_BATCH_SIZE", raising=False)
    config = tmp_path / "train.yaml"
    config.write_text(
        "\n".join(
            [
                "training_backend: ms_swift",
                "num_gpus: 2",
                "micro_batch_size: 3",
                "eval_batch_size: 2",
                "gradient_accumulation_steps: 2",
                "steps_per_generation: 2",
                "rollout_n: 2",
                "learning_rate: 1.0e-5",
                "scale_learning_rate_with_micro_batch: true",
            ]
        )
        + "\n"
    )

    result = _launch_stage(stage, config, dry_run=True)

    assert result["environment"]["MICRO_BATCH_SIZE"] == "3"
    assert result["environment"]["LR"] == "0.00003"


def test_explicit_lr_environment_override_is_not_scaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LR", "7e-6")
    monkeypatch.delenv("MICRO_BATCH_SIZE", raising=False)
    config = tmp_path / "sft.yaml"
    config.write_text(
        "training_backend: ms_swift\n"
        "micro_batch_size: 2\n"
        "learning_rate: 1.0e-5\n"
        "scale_learning_rate_with_micro_batch: true\n"
    )

    result = _launch_stage("sft", config, dry_run=True)

    assert result["environment"]["LR"] == "7e-6"


def test_lr_scaling_requires_boolean_flag(tmp_path: Path) -> None:
    config = tmp_path / "sft.yaml"
    config.write_text(
        "training_backend: ms_swift\n"
        "learning_rate: 1.0e-5\n"
        "scale_learning_rate_with_micro_batch: 'yes'\n"
    )

    with pytest.raises(
        ValueError, match="scale_learning_rate_with_micro_batch must be a boolean"
    ):
        _launch_stage("sft", config, dry_run=True)


def test_launch_stage_rejects_invalid_vllm_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_MODE", raising=False)
    config = tmp_path / "grpo.yaml"
    config.write_text("training_backend: ms_swift\nvllm_mode: invalid\n")

    with pytest.raises(ValueError, match="VLLM_MODE must be server or colocate"):
        _launch_stage("solver-grpo", config, dry_run=True)


def test_rl_export_uses_backend_neutral_name() -> None:
    args = build_parser().parse_args(
        [
            "data",
            "export-rl",
            "--input",
            "tasks.parquet",
            "--output",
            "rl.parquet",
        ]
    )
    assert args.action == "export-rl"
