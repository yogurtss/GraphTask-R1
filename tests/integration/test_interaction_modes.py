import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from graphtask_r1.data import select_graphscript_tasks
from graphtask_r1.data.seeds import export_questioner_task_seeds, sample_questioner_seeds
from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import InMemoryGraphBackend, toy_graph
from graphtask_r1.graphscript import program_to_graphscript
from graphtask_r1.schema import (
    Entity,
    Hop,
    QueryAttribute,
    QueryRelation,
    RelationInfo,
    TaskCertificate,
    TaskProposal,
    TaskTrainingRecord,
    Triple,
)
from graphtask_r1.training.ms_swift_data import convert_rl_row
from graphtask_r1.training.opponent import FrozenSolverService
from graphtask_r1.training.questioner_sampling import select_questioner_tasks
from graphtask_r1.training.relations import build_relation_catalog, load_relation_catalog
from graphtask_r1.training.rl_dataset import export_role_dataset
from graphtask_r1.training.selfplay import (
    SelfPlayConfig,
    _adapter_from_checkpoint,
    _assemble_dataset,
    _commands,
    _prepare_validation_dataset,
    _validate_merged_model,
    load_selfplay_config,
    run_self_play,
)
from graphtask_r1.training.sft_dataset import (
    combine_sft_datasets,
    export_questioner_sft_dataset,
    export_sft_dataset,
)
from graphtask_r1.utils import write_json, write_records


def _task():
    program = Hop(
        input=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        relation="located_in",
    )
    return certify_proposal(
        TaskProposal(topic_entities=("alice",), program=program),
        toy_graph(),
        graph_snapshot="toy-v1",
    )


def _catalog() -> tuple[RelationInfo, ...]:
    return (
        RelationInfo(relation_id="located_in", label="located in"),
        RelationInfo(relation_id="works_at", label="works at"),
    )


def _multi_root_task() -> TaskCertificate:
    return certify_proposal(
        TaskProposal(
            topic_entities=("acme", "alice"),
            program=QueryRelation(
                subject=Entity(entity_id="alice"),
                object=Entity(entity_id="acme"),
            ),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
    )


def test_graphscript_datasets_use_single_turn_without_tools(tmp_path: Path) -> None:
    task = _task()
    sft_path = tmp_path / "code_sft.parquet"
    rl_path = tmp_path / "code_rl.parquet"
    assert (
        export_sft_dataset(
            [task], sft_path, interaction_mode="graphscript", relation_catalog=_catalog()
        )
        == 2
    )
    sft_rows = pq.read_table(sft_path).to_pylist()
    for row in sft_rows:
        output = json.loads(row["messages"][-1]["content"])
        assert output["version"] == "0.1"
        assert row["interaction_mode"] == "graphscript"
    assert (
        export_role_dataset(
            [task], rl_path, interaction_mode="graphscript", relation_catalog=_catalog()
        )
        == 2
    )
    rl_rows = pq.read_table(rl_path).to_pylist()
    assert all("agent_name" not in row for row in rl_rows)
    assert all("tools_kwargs" not in row for row in rl_rows)
    assert all(row["extra_info"]["interaction_mode"] == "graphscript" for row in rl_rows)


def test_graphscript_v02_sft_uses_question_only_and_full_program_ir(
    tmp_path: Path,
) -> None:
    path = tmp_path / "code_v02_sft.parquet"
    assert (
        export_sft_dataset(
            [_task()],
            path,
            include_questioner=False,
            interaction_mode="graphscript",
            graphscript_version="0.2",
            relation_catalog=_catalog(),
        )
        == 1
    )
    row = pq.read_table(path).to_pylist()[0]
    assert row["graphscript_version"] == "0.2"
    assert "Topic entities" not in row["messages"][1]["content"]
    output = json.loads(row["messages"][-1]["content"])
    assert output["version"] == "0.2"
    assert output["ops"][0] == {
        "op": "resolve_entity",
        "query": "Alice",
        "match": "exact",
        "limit": 1,
        "out": "h0",
    }


def test_graphscript_v02_grpo_uses_same_question_only_operator_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "code_v02_grpo.parquet"
    assert (
        export_role_dataset(
            [_task()],
            path,
            include_questioner=False,
            interaction_mode="graphscript",
            graphscript_version="0.2",
            relation_catalog=_catalog(),
            program_profile="graphscript_v0_2",
        )
        == 1
    )
    row = pq.read_table(path).to_pylist()[0]
    assert row["extra_info"]["graphscript_version"] == "0.2"
    assert "search_passage" in row["extra_info"]["operator_set"]
    assert "Topic entities" not in row["prompt"][1]["content"]
    assert "GraphScript v0.2" in row["prompt"][0]["content"]
    assert "tools_kwargs" not in row


def test_graphscript_exports_write_multiple_bounded_parquet_batches(tmp_path: Path) -> None:
    tasks = [_task()] * 300
    sft_path = tmp_path / "batched_sft.parquet"
    rl_path = tmp_path / "batched_rl.parquet"

    assert (
        export_sft_dataset(
            tasks,
            sft_path,
            include_questioner=False,
            interaction_mode="graphscript",
            graphscript_version="0.2",
            relation_catalog=_catalog(),
        )
        == 300
    )
    assert (
        export_role_dataset(
            tasks,
            rl_path,
            include_questioner=False,
            interaction_mode="graphscript",
            graphscript_version="0.2",
            relation_catalog=_catalog(),
            program_profile="graphscript_v0_2",
        )
        == 300
    )
    assert pq.ParquetFile(sft_path).metadata.num_row_groups == 2
    assert pq.ParquetFile(rl_path).metadata.num_row_groups == 2


def test_tool_dataset_is_backend_neutral_and_adapted_at_load_time(tmp_path: Path) -> None:
    path = tmp_path / "tool_rl.parquet"
    assert (
        export_role_dataset(
            [_task()], path, interaction_mode="tool", relation_catalog=_catalog()
        )
        == 2
    )
    rows = pq.read_table(path).to_pylist()
    assert all("agent_name" not in row for row in rows)
    assert all("tools_kwargs" not in row for row in rows)
    adapted = [convert_rl_row(row) for row in rows]
    assert all(row["tools"] for row in adapted)


def test_tool_sft_serializes_query_arguments_as_json(tmp_path: Path) -> None:
    path = tmp_path / "tool_sft.parquet"

    assert (
        export_sft_dataset(
            [_task()], path, interaction_mode="tool", relation_catalog=_catalog()
        )
        == 2
    )

    solver = next(row for row in pq.read_table(path).to_pylist() if row["role"] == "solver")
    tool_call = next(
        message["tool_calls"][0] for message in solver["messages"] if message.get("tool_calls")
    )
    arguments = tool_call["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments)["query"]["steps"][0]["op"] == "hop"


def test_questioner_only_tool_sft_remains_backend_neutral(tmp_path: Path) -> None:
    task = _task().model_copy(update={"graph_snapshot": "unconfigured-snapshot"})
    path = tmp_path / "questioner_tool_sft.parquet"

    assert (
        export_sft_dataset(
            [task],
            path,
            include_questioner=True,
            include_solver=False,
            interaction_mode="tool",
        )
        == 1
    )
    assert pq.read_table(path)["role"].to_pylist() == ["questioner"]


def test_graphscript_selfplay_dry_run_selects_mode(tmp_path: Path) -> None:
    config = tmp_path / "graphscript.yaml"
    config.write_text(
        "\n".join(
            [
                "model_path: Qwen/Qwen3-4B-Instruct-2507",
                f"initial_adapter: {tmp_path / 'adapter'}",
                f"base_tasks: {tmp_path / 'tasks.parquet'}",
                f"val_data: {tmp_path / 'val.parquet'}",
                f"questioner_seeds: {tmp_path / 'seeds.parquet'}",
                "graph_snapshot: toy-v1",
                "rounds: 1",
                f"relation_catalog: {tmp_path / 'relations.json'}",
                "interaction_mode: graphscript",
                'graphscript_version: "0.1"',
                "program_profile: graphscript_v0_1",
            ]
        )
        + "\n"
    )
    result = run_self_play(config, tmp_path / "run", resume=False, dry_run=True)
    plan = result["plans"][0]
    assert plan["interaction_mode"] == "graphscript"
    assert "--interaction-mode" in plan["commands"]["opponent"]
    mode_index = plan["commands"]["opponent"].index("--interaction-mode")
    assert plan["commands"]["opponent"][mode_index + 1] == "graphscript"
    merged_model = str((tmp_path / "run" / "round_001" / "opponent_merged").resolve())
    assert plan["commands"]["merge"] == [
        "swift",
        "export",
        "--model",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--model_type",
        "qwen3",
        "--adapters",
        str((tmp_path / "adapter").resolve()),
        "--train_type",
        "lora",
        "--torch_dtype",
        "bfloat16",
        "--load_args",
        "false",
        "--merge_lora",
        "true",
        "--output_dir",
        merged_model,
    ]
    assert plan["merged_opponent_model"] == merged_model
    assert plan["commands"]["merge"][
        plan["commands"]["merge"].index("--load_args") + 1
    ] == "false"
    sglang_command = plan["commands"]["sglang"]
    assert sglang_command[sglang_command.index("--model-path") + 1] == merged_model
    assert "--enable-lora" not in sglang_command
    assert "--lora-paths" not in sglang_command
    opponent_command = plan["commands"]["opponent"]
    assert opponent_command[opponent_command.index("--model") + 1] == merged_model
    assert plan["actor_gpus"] == "0,1,2"
    assert plan["opponent_gpus"] == "3"
    assert plan["train_environment"]["TRAIN_CUDA_VISIBLE_DEVICES"] == "0,1,2"
    assert plan["train_environment"]["NUM_GPUS"] == "3"
    assert plan["train_environment"]["MAX_COMPLETION_LENGTH"] == "4096"
    assert plan["train_environment"]["VLLM_MAX_MODEL_LEN"] == "16384"
    assert plan["train_environment"]["VLLM_GPU_MEMORY_UTILIZATION"] == "0.6"
    assert plan["train_environment"]["VLLM_SLEEP_LEVEL"] == "1"
    assert plan["train_environment"]["DEEPSPEED"] == "none"
    assert plan["train_environment"]["RL_ALGORITHM"] == "grpo"
    assert plan["train_environment"]["SAVE_STEPS"] == "20"
    assert plan["train_environment"]["SAVE_TOTAL_LIMIT"] == "2"
    assert "EVAL_ROLLOUT_N" not in plan["train_environment"]
    assert plan["train_environment"]["VLLM_MODE"] == "colocate"
    assert plan["deepspeed"] == "none"
    assert plan["rl_algorithm"] == "grpo"
    assert plan["rollout_budget"] == {
        "questioner_prompts": 256,
        "solver_prompts": 256,
        "actor_completions_upper_bound": 2_048,
        "opponent_completions_upper_bound": 4_096,
    }


def test_selfplay_validates_merged_opponent_artifacts(tmp_path: Path) -> None:
    merged = tmp_path / "opponent_merged"
    merged.mkdir()
    with pytest.raises(RuntimeError, match="config.json"):
        _validate_merged_model(merged)

    (merged / "config.json").write_text("{}")
    (merged / "tokenizer_config.json").write_text("{}")
    (merged / "model-00001-of-00002.safetensors").touch()

    _validate_merged_model(merged)


def test_interaction_task_selection_and_catalog_are_deterministic(tmp_path: Path) -> None:
    selected_task = _task()
    one_hop = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=Hop(input=Entity(entity_id="alice"), relation="friend"),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
    )
    output = tmp_path / "interaction.parquet"
    metrics = select_graphscript_tasks([selected_task, one_hop], output, backend=toy_graph())
    assert metrics == {"input": 2, "selected": 1, "rejected": 1}
    rows = pq.read_table(output)["record_json"].to_pylist()
    restored = [TaskCertificate.model_validate(json.loads(value)) for value in rows]
    catalog_path = tmp_path / "relations.json"
    relations = build_relation_catalog(restored, toy_graph(), catalog_path)
    assert [value.relation_id for value in relations] == ["located_in", "works_at"]
    assert load_relation_catalog(catalog_path) == relations


def test_graph_schema_catalog_is_independent_of_task_sample(tmp_path: Path) -> None:
    catalog_path = tmp_path / "graph_relations.json"

    relations = build_relation_catalog(
        [_task()],
        toy_graph(),
        catalog_path,
        include_graph_schema=True,
    )

    assert {relation.relation_id for relation in relations} == {
        "age",
        "country",
        "friend",
        "friend_of_friend",
        "located_in",
        "works_at",
    }


def test_selfplay_defaults_use_kqapro_graphscript_profile() -> None:
    config = SelfPlayConfig.model_validate(
        {
            "initial_adapter": "adapter",
            "base_tasks": "tasks.parquet",
            "val_data": "val.parquet",
            "questioner_seeds": "seeds.parquet",
        }
    )
    assert config.graph_snapshot == "kqapro-v1"
    assert config.selfplay_variant == "legacy"
    assert config.interaction_mode == "graphscript"
    assert config.graphscript_version == "0.3"
    assert config.program_profile == "graphscript_v0_3"
    assert config.questioner_episodes == 256
    assert config.solver_episodes == 256
    assert config.opponent_samples == 4
    assert config.actor_gpus == "0,1,2"
    assert config.opponent_gpus == "3"
    assert config.allow_gpu_overlap is False
    assert config.use_vllm is True
    assert config.opponent_backend == "sglang"
    assert config.max_completion_tokens == 4_096
    assert config.vllm_max_model_len == 16_384
    assert config.vllm_gpu_memory_utilization == 0.6
    assert config.vllm_sleep_level == 1
    assert config.deepspeed == "none"
    assert config.rl_algorithm == "grpo"
    assert config.micro_batch_size == 4
    assert config.eval_batch_size == 8
    assert config.gradient_accumulation_steps == 2
    assert config.steps_per_generation == 4
    assert config.rollout_n == 4
    assert config.save_steps == 20
    assert config.save_total_limit == 2
    assert config.validation_samples == 256


def test_selfplay_rejects_overlapping_actor_and_opponent_gpus() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "actor_gpus": "0,1",
                "opponent_gpus": "1,2",
            }
        )


def test_selfplay_allows_explicit_single_gpu_transformers_smoke() -> None:
    config = SelfPlayConfig.model_validate(
        {
            "initial_adapter": "adapter",
            "base_tasks": "tasks.parquet",
            "val_data": "val.parquet",
            "questioner_seeds": "seeds.parquet",
            "actor_gpus": "0",
            "opponent_gpus": "0",
            "allow_gpu_overlap": True,
            "use_vllm": False,
            "opponent_backend": "transformers",
            "opponent_samples": 1,
            "micro_batch_size": 1,
            "eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "steps_per_generation": 2,
            "rollout_n": 2,
        }
    )

    commands = _commands(
        config,
        adapter=Path("adapter"),
        archive_path=Path("archive.sqlite"),
        mixed_data=Path("mixed.parquet"),
        round_dir=Path("round"),
    )
    assert config.actor_gpus == "0"
    assert config.opponent_gpus == "0"
    assert config.allow_gpu_overlap is True
    assert config.use_vllm is False
    assert config.opponent_backend == "transformers"
    assert commands["sglang"] == []
    assert "--local-model" in commands["opponent"]
    assert "--model-url" not in commands["opponent"]


@pytest.mark.parametrize(
    ("use_vllm", "opponent_backend"),
    [(True, "transformers"), (False, "sglang")],
)
def test_selfplay_rejects_unsafe_single_gpu_overlap(
    use_vllm: bool, opponent_backend: str
) -> None:
    with pytest.raises(ValueError, match="overlapping GPUs require"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "actor_gpus": "0",
                "opponent_gpus": "0",
                "allow_gpu_overlap": True,
                "use_vllm": use_vllm,
                "opponent_backend": opponent_backend,
                "opponent_samples": 1,
            }
        )


def test_selfplay_reserves_vllm_context_for_the_prompt() -> None:
    with pytest.raises(ValueError, match="must exceed max_completion_tokens"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "max_completion_tokens": 4_096,
                "vllm_max_model_len": 4_096,
            }
        )


def test_selfplay_requires_mixture_ratios_to_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "base_ratio": 0.8,
                "archive_ratio": 0.3,
                "new_ratio": 0.0,
            }
        )


def test_selfplay_retry_selects_only_new_highest_checkpoint(tmp_path: Path) -> None:
    stale = tmp_path / "v0" / "checkpoint-99" / "adapter_model.safetensors"
    fresh_low = tmp_path / "v1" / "checkpoint-1" / "adapter_model.safetensors"
    fresh_high = tmp_path / "v1" / "checkpoint-2" / "adapter_model.safetensors"
    for path in (stale, fresh_low, fresh_high):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    selected = _adapter_from_checkpoint(
        tmp_path, known_adapters=frozenset({stale.resolve()})
    )

    assert selected == fresh_high.parent


def test_default_selfplay_file_uses_kqapro_snapshot() -> None:
    config_path = Path(__file__).parents[2] / "configs/training/selfplay.yaml"
    config = load_selfplay_config(config_path)
    assert config.graph_snapshot == "kqapro-v1"
    assert config.deepspeed == "zero2"
    assert config.rl_algorithm == "reinforce_plus_plus"
    assert config.questioner_episodes == 192
    assert config.solver_episodes == 320
    assert config.eval_batch_size == 4
    assert config.validation_samples == 256
    assert config.selfplay_variant == "legacy"


def test_frontier_v2_config_is_isolated_from_legacy() -> None:
    root = Path(__file__).parents[2]
    legacy = load_selfplay_config(root / "configs/training/selfplay.yaml")
    frontier_v2 = load_selfplay_config(root / "configs/training/selfplay_frontier_v2.yaml")

    assert legacy.selfplay_variant == "legacy"
    assert legacy.opponent_samples == 4
    assert frontier_v2.selfplay_variant == "frontier_v2"
    assert frontier_v2.opponent_samples == 8
    assert frontier_v2.frontier_target_start == 0.55
    assert frontier_v2.frontier_target_end == 0.40
    commands = _commands(
        frontier_v2,
        adapter=Path("adapter"),
        archive_path=Path("archive.sqlite"),
        mixed_data=Path("mixed.parquet"),
        round_dir=Path("round"),
    )
    assert "--candidate-archive" in commands["opponent"]
    assert "--cache-evaluations" in commands["opponent"]


def test_validation_subset_is_bounded_deterministic_and_replayable(tmp_path: Path) -> None:
    source = tmp_path / "val.parquet"
    pq.write_table(pa.table({"row_id": list(range(20))}), source)

    first = _prepare_validation_dataset(source, tmp_path / "run-a", max_samples=5, seed=42)
    second = _prepare_validation_dataset(source, tmp_path / "run-b", max_samples=5, seed=42)

    assert first["total_rows"] == 20
    assert first["selected_rows"] == 5
    assert first["selected_indices"] == second["selected_indices"]
    assert pq.read_table(first["output"])["row_id"].to_pylist() == pq.read_table(
        second["output"]
    )["row_id"].to_pylist()
    manifest = json.loads((tmp_path / "run-a" / "validation_sample.json").read_text())
    assert manifest == first


@pytest.mark.parametrize(
    "stage",
    ["none", "zero0", "zero1", "zero2", "zero3", "zero2_offload", "zero3_offload"],
)
def test_selfplay_accepts_supported_deepspeed_stages(stage: str) -> None:
    config = SelfPlayConfig.model_validate(
        {
            "initial_adapter": "adapter",
            "base_tasks": "tasks.parquet",
            "val_data": "val.parquet",
            "questioner_seeds": "seeds.parquet",
            "deepspeed": stage,
        }
    )

    assert config.deepspeed == stage


def test_selfplay_rejects_unknown_deepspeed_stage() -> None:
    with pytest.raises(ValueError, match="deepspeed"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "deepspeed": "zero4",
            }
        )


@pytest.mark.parametrize("algorithm", ["grpo", "reinforce_plus_plus"])
def test_selfplay_accepts_supported_rl_algorithms(algorithm: str) -> None:
    config = SelfPlayConfig.model_validate(
        {
            "initial_adapter": "adapter",
            "base_tasks": "tasks.parquet",
            "val_data": "val.parquet",
            "questioner_seeds": "seeds.parquet",
            "rl_algorithm": algorithm,
        }
    )

    assert config.rl_algorithm == algorithm


def test_selfplay_rejects_unknown_rl_algorithm() -> None:
    with pytest.raises(ValueError, match="rl_algorithm"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "rl_algorithm": "reinforce",
            }
        )


def test_selfplay_validates_eval_batch_against_training_rollout_n() -> None:
    with pytest.raises(ValueError, match="evaluation batch must be divisible by rollout_n"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "actor_gpus": "0,1,2",
                "eval_batch_size": 2,
                "rollout_n": 4,
            }
        )


def test_comparison_profile_requires_relation_catalog() -> None:
    with pytest.raises(ValueError, match="comparison profile requires relation_catalog"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "interaction_mode": "graphscript",
                "graphscript_version": "0.1",
                "program_profile": "graphscript_v0_1",
            }
        )


def test_selection_rejects_task_that_only_full_execution_can_answer(tmp_path: Path) -> None:
    backend = InMemoryGraphBackend(
        [
            Triple(subject="seed", relation="r1", object="a"),
            Triple(subject="seed", relation="r1", object="b"),
            Triple(subject="b", relation="r2", object="gold"),
        ]
    )
    program = Hop(
        input=Hop(input=Entity(entity_id="seed"), relation="r1"),
        relation="r2",
    )
    task = certify_proposal(
        TaskProposal(topic_entities=("seed",), program=program),
        backend,
        graph_snapshot="bounded-test-v1",
    )
    output = tmp_path / "bounded.parquet"
    metrics = select_graphscript_tasks(
        [task],
        output,
        backend=backend,
        max_follow_limit=1,
        max_edge_visits=2,
    )
    assert metrics == {"input": 1, "selected": 0, "rejected": 1}
    rejection = json.loads(
        pq.read_table(output.with_name("bounded_rejections.parquet"))["record_json"][0].as_py()
    )
    assert rejection["reason_code"] == "EMPTY_RESULT"


def test_graphscript_export_rejects_incomplete_catalog(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing program relations: located_in"):
        export_sft_dataset(
            [_task()],
            tmp_path / "bad.parquet",
            interaction_mode="graphscript",
            relation_catalog=(RelationInfo(relation_id="works_at", label="works at"),),
        )


class _ScriptedOpponent(FrozenSolverService):
    def __init__(self, tmp_path: Path, responses: list[dict[str, Any]]) -> None:
        super().__init__(
            model_url="http://unused",
            model="scripted",
            archive_path=tmp_path / "opponent.sqlite",
            relation_catalog=_catalog(),
            max_edge_visits=10,
        )
        self.responses = iter(responses)

    async def _completion(
        self, messages: list[dict[str, Any]], *, use_tools: bool
    ) -> dict[str, Any]:
        del messages, use_tools
        return next(self.responses)


class _SeededCachedOpponent(FrozenSolverService):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(
            model_url="http://unused",
            model="seeded-scripted",
            archive_path=tmp_path / "archive.sqlite",
            candidate_archive_path=tmp_path / "candidates.sqlite",
            cache_evaluations=True,
            interaction_mode="graphscript",
            graphscript_version="0.1",
            relation_catalog=_catalog(),
            max_edge_visits=10,
        )
        self.seeds: list[int | None] = []

    async def _completion(
        self,
        messages: list[dict[str, Any]],
        *,
        use_tools: bool,
        seed: int | None = None,
    ) -> dict[str, Any]:
        del messages, use_tools
        self.seeds.append(seed)
        script = program_to_graphscript(_task().program)
        return {"role": "assistant", "content": script.model_dump_json(by_alias=True)}


def test_frontier_v2_opponent_seeds_caches_and_stages_without_mutating_archive(
    tmp_path: Path,
) -> None:
    service = _SeededCachedOpponent(tmp_path)
    task = _task()
    payload = {
        "proposal": TaskProposal(
            topic_entities=("alice",), program=task.program
        ).model_dump(mode="json"),
        "graph_snapshot": "toy-v1",
        "samples": 2,
        "round": 1,
        "interaction_mode": "graphscript",
        "graphscript_version": "0.1",
        "allowed_relations": ["works_at", "located_in"],
        "max_follow_limit": 100,
        "max_edge_visits": 10,
        "seed": 42,
    }

    async def evaluate_twice() -> tuple[dict[str, Any], dict[str, Any]]:
        return await service.evaluate(payload), await service.evaluate(payload)

    first, second = asyncio.run(evaluate_twice())

    assert first == second
    assert first["pass_rate"] == 1.0
    assert len(service.seeds) == 2
    assert len(set(service.seeds)) == 2
    assert all(seed is not None for seed in service.seeds)
    assert service.archive.all() == []
    assert service.candidate_archive is not None
    assert len(service.candidate_archive.all()) == 1
    service.archive.close()
    service.candidate_archive.close()


def _tool_call(name: str, arguments: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_tool_opponent_is_confined_to_observed_frontier(tmp_path: Path) -> None:
    legacy_invalid = _ScriptedOpponent(
        tmp_path,
        [_tool_call("graph_search", {"entity_ids": ["cara"]}, 0)],
    )
    legacy_rejected = asyncio.run(
        legacy_invalid.rollout(
            _task(),
            toy_graph(),
            allowed_relations=("works_at", "located_in"),
            max_edge_visits=10,
        )
    )
    assert legacy_rejected["passed"] == 0.0
    assert "program_parse" not in legacy_rejected

    invalid = _ScriptedOpponent(
        tmp_path,
        [
            _tool_call("graph_search", {"entity_ids": ["cara"]}, 0),
            {"role": "assistant", "content": '<answer>["wrong"]</answer>'},
        ],
    )
    rejected = asyncio.run(
        invalid.rollout(
            _task(),
            toy_graph(),
            allowed_relations=("works_at", "located_in"),
            max_edge_visits=10,
            recover_invalid_tool_calls=True,
        )
    )
    assert rejected["passed"] == 0.0
    assert rejected["edge_visits"] == 0.0
    assert rejected["tool_calls"] == 1.0
    assert rejected["invalid_tool_calls"] == 1.0
    assert rejected["program_parse"] == 1.0

    valid = _ScriptedOpponent(
        tmp_path,
        [
            _tool_call(
                "graph_search",
                {
                    "entity_ids": ["alice"],
                    "direction": "out",
                    "relation_ids": ["works_at"],
                },
                0,
            ),
            _tool_call(
                "graph_search",
                {
                    "entity_ids": ["acme"],
                    "direction": "out",
                    "relation_ids": ["located_in"],
                },
                1,
            ),
            {"role": "assistant", "content": '<answer>["paris"]</answer>'},
        ],
    )
    accepted = asyncio.run(
        valid.rollout(
            _task(),
            toy_graph(),
            allowed_relations=("works_at", "located_in"),
            max_edge_visits=10,
        )
    )
    assert accepted["passed"] == 1.0
    assert accepted["edge_visits"] == 2.0


def test_tool_opponent_parses_literal_answers_from_gold_kind(tmp_path: Path) -> None:
    literal_task = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=QueryAttribute(input=Entity(entity_id="alice"), attribute="age"),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
    )
    value = literal_task.gold_answers.answers[0].value
    service = _ScriptedOpponent(
        tmp_path,
        [{"role": "assistant", "content": f"<answer>{json.dumps([value])}</answer>"}],
    )

    result = asyncio.run(service.rollout(literal_task, toy_graph()))

    assert literal_task.gold_answers.answers[0].kind == "literal"
    assert result["passed"] == 1.0
    assert result["program_parse"] == 1.0
    assert result["program_executable"] == 1.0


def test_graphscript_selfplay_assembles_mixed_single_turn_dataset(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.parquet"
    seeds_path = tmp_path / "seeds.parquet"
    catalog_path = tmp_path / "relations.json"
    # Production BASE_TASKS points at the compact audit training view, not a full
    # TaskCertificate with witnesses and provenance.
    training_task = TaskTrainingRecord.model_validate(_task().model_dump(mode="json"))
    write_records(tasks_path, [training_task.model_dump(mode="json")])
    write_json(catalog_path, [value.model_dump(mode="json") for value in _catalog()])
    sample_questioner_seeds(
        "toy-v1",
        seeds_path,
        count=1,
        seed=42,
        min_degree=1,
        interaction_mode="graphscript",
        graphscript_version="0.1",
        relation_catalog=_catalog(),
    )
    config = SelfPlayConfig(
        initial_adapter="adapter",
        base_tasks=tasks_path,
        val_data=tmp_path / "val.parquet",
        questioner_seeds=seeds_path,
        graph_snapshot="toy-v1",
        solver_episodes=1,
        interaction_mode="graphscript",
        graphscript_version="0.1",
        relation_catalog=catalog_path,
        relation_catalogs={"toy-v1": catalog_path},
        program_profile="graphscript_v0_1",
    )
    output = tmp_path / "mixed.parquet"
    counts = _assemble_dataset(
        config,
        tmp_path / "archive.sqlite",
        output,
        round_index=1,
        opponent_url="http://127.0.0.1:18080",
    )
    assert counts == {"questioner": 1, "solver": 1, "total": 2}
    table = pq.read_table(output)
    assert table.num_rows == 2
    assert "agent_name" not in table.column_names
    assert "tools_kwargs" not in table.column_names


def test_selfplay_bounds_questioner_rows_per_round(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.parquet"
    seeds_path = tmp_path / "seeds.parquet"
    catalog_path = tmp_path / "relations.json"
    training_task = TaskTrainingRecord.model_validate(_task().model_dump(mode="json"))
    write_records(tasks_path, [training_task.model_dump(mode="json")])
    write_json(catalog_path, [value.model_dump(mode="json") for value in _catalog()])
    sample_questioner_seeds(
        "toy-v1",
        seeds_path,
        count=3,
        seed=42,
        min_degree=1,
        interaction_mode="graphscript",
        graphscript_version="0.1",
        relation_catalog=_catalog(),
    )
    config = SelfPlayConfig(
        initial_adapter="adapter",
        base_tasks=tasks_path,
        val_data=tmp_path / "val.parquet",
        questioner_seeds=seeds_path,
        graph_snapshot="toy-v1",
        questioner_episodes=1,
        solver_episodes=1,
        interaction_mode="graphscript",
        graphscript_version="0.1",
        relation_catalog=catalog_path,
        relation_catalogs={"toy-v1": catalog_path},
        program_profile="graphscript_v0_1",
    )

    counts = _assemble_dataset(
        config,
        tmp_path / "archive.sqlite",
        tmp_path / "mixed.parquet",
        round_index=1,
        opponent_url="http://127.0.0.1:18080",
    )

    assert counts == {"questioner": 1, "solver": 1, "total": 2}


def test_frontier_v2_writes_independent_role_datasets_and_reward_contract(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.parquet"
    seeds_path = tmp_path / "seeds.parquet"
    catalog_path = tmp_path / "relations.json"
    write_records(
        tasks_path,
        [TaskTrainingRecord.model_validate(_task().model_dump(mode="json")).model_dump()],
    )
    write_json(catalog_path, [value.model_dump(mode="json") for value in _catalog()])
    sample_questioner_seeds(
        "toy-v1",
        seeds_path,
        count=1,
        seed=42,
        min_degree=1,
        interaction_mode="graphscript",
        graphscript_version="0.1",
        relation_catalog=_catalog(),
    )
    config = SelfPlayConfig(
        initial_adapter="adapter",
        base_tasks=tasks_path,
        val_data=tmp_path / "val.parquet",
        questioner_seeds=seeds_path,
        graph_snapshot="toy-v1",
        selfplay_variant="frontier_v2",
        rounds=3,
        questioner_episodes=2,
        solver_episodes=2,
        base_ratio=1.0,
        archive_ratio=0.0,
        new_ratio=0.0,
        interaction_mode="graphscript",
        graphscript_version="0.1",
        relation_catalog=catalog_path,
        relation_catalogs={"toy-v1": catalog_path},
        program_profile="graphscript_v0_1",
    )

    output = tmp_path / "mixed.parquet"
    counts = _assemble_dataset(
        config,
        tmp_path / "archive.sqlite",
        output,
        round_index=2,
        opponent_url="http://127.0.0.1:18080",
    )

    assert counts == {"questioner": 2, "solver": 2, "total": 4}
    questioner_rows = pq.read_table(tmp_path / "questioner.parquet").to_pylist()
    assert len(questioner_rows) == 2
    assert all(
        row["extra_info"]["questioner_reward_variant"] == "frontier_v2"
        for row in questioner_rows
    )
    assert all(row["extra_info"]["opponent_seed"] == 42 for row in questioner_rows)
    assert all(row["extra_info"]["frontier_target"] == 0.5 for row in questioner_rows)
    assert pq.ParquetFile(tmp_path / "solver.parquet").metadata.num_rows == 2


def test_curriculum_prompt_hint_does_not_narrow_execution_relations(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.parquet"
    seeds_path = tmp_path / "seeds.parquet"
    catalog_path = tmp_path / "relations.json"
    write_records(
        tasks_path,
        [TaskTrainingRecord.model_validate(_task().model_dump(mode="json")).model_dump()],
    )
    write_json(catalog_path, [value.model_dump(mode="json") for value in _catalog()])
    sample_questioner_seeds(
        "toy-v1",
        seeds_path,
        count=1,
        seed=42,
        min_degree=1,
        interaction_mode="graphscript",
        graphscript_version="0.1",
        relation_catalog=_catalog(),
    )
    config = SelfPlayConfig(
        initial_adapter="adapter",
        base_tasks=tasks_path,
        val_data=tmp_path / "val.parquet",
        questioner_seeds=seeds_path,
        graph_snapshot="toy-v1",
        selfplay_variant="curriculum_v3",
        rounds=3,
        questioner_episodes=1,
        solver_episodes=1,
        base_ratio=1.0,
        archive_ratio=0.0,
        new_ratio=0.0,
        interaction_mode="graphscript",
        graphscript_version="0.1",
        relation_catalog=catalog_path,
        relation_catalogs={"toy-v1": catalog_path},
        program_profile="graphscript_v0_1",
    )

    _assemble_dataset(
        config,
        tmp_path / "archive.sqlite",
        tmp_path / "mixed.parquet",
        round_index=1,
        opponent_url="http://127.0.0.1:18080",
    )

    row = pq.read_table(tmp_path / "questioner.parquet").to_pylist()[0]
    info = row["extra_info"]
    assert set(info["allowed_relations"]) == {"works_at", "located_in"}
    assert info["observed_relation_ids"] == ["works_at"]
    assert "- works_at: works at" in row["prompt"][1]["content"]
    assert "- located_in: located in" not in row["prompt"][1]["content"]


def test_kqapro_questioner_seeds_default_to_graphscript_v03(tmp_path: Path) -> None:
    output = tmp_path / "seeds.parquet"
    sample_questioner_seeds(
        "toy-v1",
        output,
        count=1,
        seed=42,
        min_degree=1,
        interaction_mode="graphscript",
        relation_catalog=_catalog(),
    )

    row = pq.read_table(output).to_pylist()[0]
    assert "structured graph self-play" in row["prompt"][0]["content"]
    assert row["extra_info"]["graphscript_version"] == "0.3"
    assert row["extra_info"]["program_profile"] == "graphscript_v0_3"
    assert "filter_qualifier" in row["extra_info"]["operator_set"]
    assert row["extra_info"]["seed_context"]
    system = row["prompt"][0]["content"]
    user = row["prompt"][1]["content"]
    assert '{"version":"0.3","ops":[...]}' in system
    assert "never use all_entities" in system
    assert "all_entities(max_results" not in system
    assert "required_root_op" in user
    assert '"match":"id"' in user
    assert "observed_outgoing_relation_ids" in user


def test_questioner_sft_has_independent_exact_count_and_combines_with_solver(
    tmp_path: Path,
) -> None:
    tasks = [
        TaskTrainingRecord.model_validate(
            _task().model_copy(update={"task_id": f"task-{index}"}).model_dump(mode="json")
        )
        for index in range(5)
    ]
    questioner_path = tmp_path / "questioner.parquet"
    solver_path = tmp_path / "solver.parquet"
    mixed_path = tmp_path / "mixed.parquet"

    metrics = export_questioner_sft_dataset(
        tasks,
        questioner_path,
        count=2,
        seed=7,
        interaction_mode="graphscript",
        graphscript_version="0.3",
        relation_catalog=_catalog(),
    )
    assert metrics == {
        "scanned": 5,
        "requested": 2,
        "eligible": 5,
        "unique_selected": 2,
        "repeated_rows": 0,
        "shortfall": 0,
        "selected": 2,
        "seed": 7,
    }
    questioner_rows = pq.read_table(questioner_path).to_pylist()
    assert len(questioner_rows) == 2
    assert {row["role"] for row in questioner_rows} == {"questioner"}
    assert all("required_root_op" in row["messages"][1]["content"] for row in questioner_rows)
    for row in questioner_rows:
        output = json.loads(row["messages"][-1]["content"])
        assert output["ops"][0] == {
            "op": "resolve_entity",
            "query": "alice",
            "match": "id",
            "limit": 1,
            "out": "h0",
        }

    assert (
        export_sft_dataset(
            tasks,
            solver_path,
            include_questioner=False,
            interaction_mode="graphscript",
            graphscript_version="0.3",
            relation_catalog=_catalog(),
        )
        == 5
    )
    combined = combine_sft_datasets(solver_path, questioner_path, mixed_path, seed=11)
    assert combined == {
        "solver": 5,
        "questioner": 2,
        "total": 7,
        "seed": 11,
        "solver_input_rows": 5,
        "questioner_input_rows": 2,
        "solver_dropped": 0,
        "questioner_dropped": 0,
    }
    roles = pq.read_table(mixed_path)["role"].to_pylist()
    assert roles.count("solver") == 5
    assert roles.count("questioner") == 2


def test_questioner_sft_random_sample_is_deterministic_and_without_replacement(
    tmp_path: Path,
) -> None:
    tasks = [
        TaskTrainingRecord.model_validate(
            _task().model_copy(update={"task_id": f"task-{index}"}).model_dump(mode="json")
        )
        for index in range(5)
    ]
    first_path = tmp_path / "questioner-first.parquet"
    second_path = tmp_path / "questioner-second.parquet"

    first = export_questioner_sft_dataset(
        tasks,
        first_path,
        count=2,
        seed=7,
        interaction_mode="graphscript",
        graphscript_version="0.3",
        relation_catalog=_catalog(),
    )
    second = export_questioner_sft_dataset(
        tasks,
        second_path,
        count=2,
        seed=7,
        interaction_mode="graphscript",
        graphscript_version="0.3",
        relation_catalog=_catalog(),
    )

    assert first == {
        "scanned": 5,
        "requested": 2,
        "eligible": 5,
        "unique_selected": 2,
        "repeated_rows": 0,
        "shortfall": 0,
        "selected": 2,
        "seed": 7,
    }
    assert second == first
    first_ids = pq.read_table(first_path)["task_id"].to_pylist()
    second_ids = pq.read_table(second_path)["task_id"].to_pylist()
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids)) == 2


def test_questioner_sft_and_selfplay_seeds_preserve_multi_root_tasks(
    tmp_path: Path,
) -> None:
    tasks = [_task(), _multi_root_task()]
    sft_path = tmp_path / "questioner.parquet"
    seeds_path = tmp_path / "questioner-seeds.parquet"

    metrics = export_questioner_sft_dataset(
        tasks,
        sft_path,
        count=2,
        seed=7,
        interaction_mode="graphscript",
        graphscript_version="0.3",
        relation_catalog=_catalog(),
    )
    seed_metrics = export_questioner_task_seeds(
        tasks,
        seeds_path,
        count=2,
        seed=7,
        relation_catalog=_catalog(),
        graphscript_version="0.3",
    )

    assert metrics["eligible"] == 2
    assert seed_metrics["eligible"] == 2
    sft_rows = pq.read_table(sft_path).to_pylist()
    multi_sft = next(row for row in sft_rows if row["task_id"] == tasks[1].task_id)
    prompt = multi_sft["messages"][1]["content"]
    completion = json.loads(multi_sft["messages"][-1]["content"])
    resolve_ids = {
        op["query"] for op in completion["ops"] if op["op"] == "resolve_entity"
    }
    assert "Seed 1:" in prompt and "Seed 2:" in prompt
    assert '"out":"<fresh_handle>"' in prompt
    assert resolve_ids == {"alice", "acme"}

    seed_rows = pq.read_table(seeds_path).to_pylist()
    multi_seed = next(row for row in seed_rows if len(row["extra_info"]["topic_entity_ids"]) == 2)
    assert set(multi_seed["extra_info"]["topic_entity_ids"]) == {"alice", "acme"}
    serialized = json.dumps(multi_seed, ensure_ascii=False)
    assert '"program"' not in serialized
    assert '"gold_answers"' not in serialized


def test_questioner_sampling_preserves_source_root_proportions() -> None:
    single = _task()
    multi = _multi_root_task()
    tasks = [
        TaskTrainingRecord.model_validate(
            single.model_copy(update={"task_id": f"single-{index}"}).model_dump(mode="json")
        )
        for index in range(8)
    ] + [
        TaskTrainingRecord.model_validate(
            multi.model_copy(update={"task_id": f"multi-{index}"}).model_dump(mode="json")
        )
        for index in range(2)
    ]

    selected, metrics = select_questioner_tasks(tasks, count=5, seed=7)

    assert Counter(len(task.topic_entities) for task in selected) == {1: 4, 2: 1}
    assert metrics["eligible"] == 10
    assert metrics["unique_selected"] == 5
    assert metrics["distribution_total_variation"] == 0.0
    assert metrics["marginals"]["root_count"] == {
        "1": {
            "source": 8,
            "source_prevalence": 0.8,
            "final_selected": 4,
            "final_prevalence": 0.8,
        },
        "2": {
            "source": 2,
            "source_prevalence": 0.2,
            "final_selected": 1,
            "final_prevalence": 0.2,
        },
    }
    assert metrics["marginals"]["operator_presence"]["entity"] == {
        "source": 10,
        "source_prevalence": 1.0,
        "final_selected": 5,
        "final_prevalence": 1.0,
    }


def test_questioner_sft_caps_at_unique_tasks_without_repeating_by_default(
    tmp_path: Path,
) -> None:
    task = TaskTrainingRecord.model_validate(_task().model_dump(mode="json"))

    metrics = export_questioner_sft_dataset(
        [task],
        tmp_path / "questioner.parquet",
        count=2,
        seed=7,
        interaction_mode="graphscript",
        graphscript_version="0.3",
        relation_catalog=_catalog(),
    )

    assert metrics["requested"] == 2
    assert metrics["selected"] == 1
    assert metrics["repeated_rows"] == 0
    assert metrics["shortfall"] == 1
    assert pq.ParquetFile(tmp_path / "questioner.parquet").metadata.num_rows == 1


def test_combine_sft_downsamples_without_replacement_to_exact_role_ratio(
    tmp_path: Path,
) -> None:
    tasks = [
        TaskTrainingRecord.model_validate(
            _task().model_copy(update={"task_id": f"task-{index}"}).model_dump(mode="json")
        )
        for index in range(5)
    ]
    solver_path = tmp_path / "solver.parquet"
    questioner_path = tmp_path / "questioner.parquet"
    output_path = tmp_path / "mixed.parquet"
    export_sft_dataset(
        tasks,
        solver_path,
        include_questioner=False,
        interaction_mode="graphscript",
        graphscript_version="0.3",
        relation_catalog=_catalog(),
    )
    export_questioner_sft_dataset(
        tasks[:2],
        questioner_path,
        count=2,
        seed=7,
        interaction_mode="graphscript",
        graphscript_version="0.3",
        relation_catalog=_catalog(),
    )

    metrics = combine_sft_datasets(
        solver_path,
        questioner_path,
        output_path,
        seed=11,
        solver_weight=1,
        questioner_weight=1,
    )

    assert metrics["solver"] == 2
    assert metrics["questioner"] == 2
    assert metrics["solver_dropped"] == 3
    rows = pq.read_table(output_path).to_pylist()
    assert len({row["task_id"] for row in rows if row["role"] == "solver"}) == 2
