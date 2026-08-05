import json
from pathlib import Path

import pyarrow.parquet as pq

from graphtask_r1.data import select_graphscript_tasks
from graphtask_r1.data.seeds import sample_questioner_seeds
from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import Entity, Hop, RelationInfo, TaskCertificate, TaskProposal
from graphtask_r1.training.relations import build_relation_catalog, load_relation_catalog
from graphtask_r1.training.selfplay import SelfPlayConfig, _assemble_dataset, run_self_play
from graphtask_r1.training.sft_dataset import export_sft_dataset
from graphtask_r1.training.verl_dataset import export_role_dataset
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


def test_graphscript_datasets_use_single_turn_without_tools(tmp_path: Path) -> None:
    task = _task()
    sft_path = tmp_path / "code_sft.parquet"
    rl_path = tmp_path / "code_rl.parquet"
    assert export_sft_dataset(
        [task], sft_path, interaction_mode="graphscript", relation_catalog=_catalog()
    ) == 2
    sft_rows = pq.read_table(sft_path).to_pylist()
    for row in sft_rows:
        output = json.loads(row["messages"][-1]["content"])
        assert output["version"] == "0.1"
        assert row["interaction_mode"] == "graphscript"
    assert export_role_dataset(
        [task], rl_path, interaction_mode="graphscript", relation_catalog=_catalog()
    ) == 2
    rl_rows = pq.read_table(rl_path).to_pylist()
    assert all("agent_name" not in row for row in rl_rows)
    assert all("tools_kwargs" not in row for row in rl_rows)
    assert all(row["extra_info"]["interaction_mode"] == "graphscript" for row in rl_rows)


def test_tool_dataset_contract_remains_tool_agent(tmp_path: Path) -> None:
    path = tmp_path / "tool_rl.parquet"
    assert export_role_dataset([_task()], path, relation_catalog=_catalog()) == 2
    rows = pq.read_table(path).to_pylist()
    assert all(row["agent_name"] == "tool_agent" for row in rows)
    assert all(row["tools_kwargs"] for row in rows)


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
    metrics = select_graphscript_tasks([selected_task, one_hop], output)
    assert metrics == {"input": 2, "selected": 1, "rejected": 1}
    rows = pq.read_table(output)["record_json"].to_pylist()
    restored = [TaskCertificate.model_validate(json.loads(value)) for value in rows]
    catalog_path = tmp_path / "relations.json"
    relations = build_relation_catalog(restored, toy_graph(), catalog_path)
    assert [value.relation_id for value in relations] == ["located_in", "works_at"]
    assert load_relation_catalog(catalog_path) == relations


def test_selfplay_defaults_preserve_legacy_tool_profile() -> None:
    config = SelfPlayConfig.model_validate(
        {
            "initial_adapter": "adapter",
            "base_tasks": "tasks.parquet",
            "val_data": "val.parquet",
            "questioner_seeds": "seeds.parquet",
        }
    )
    assert config.interaction_mode == "tool"
    assert config.program_profile == "full"


def test_graphscript_selfplay_assembles_mixed_single_turn_dataset(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.parquet"
    seeds_path = tmp_path / "seeds.parquet"
    catalog_path = tmp_path / "relations.json"
    write_records(tasks_path, [_task().model_dump(mode="json")])
    write_json(catalog_path, [value.model_dump(mode="json") for value in _catalog()])
    sample_questioner_seeds(
        "toy-v1",
        seeds_path,
        count=1,
        seed=42,
        min_degree=1,
        interaction_mode="graphscript",
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
