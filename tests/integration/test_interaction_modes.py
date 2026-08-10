import asyncio
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from graphtask_r1.data import select_graphscript_tasks
from graphtask_r1.data.seeds import sample_questioner_seeds
from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import InMemoryGraphBackend, toy_graph
from graphtask_r1.schema import (
    Entity,
    Hop,
    RelationInfo,
    TaskCertificate,
    TaskProposal,
    Triple,
)
from graphtask_r1.training.opponent import FrozenSolverService
from graphtask_r1.training.relations import build_relation_catalog, load_relation_catalog
from graphtask_r1.training.selfplay import (
    SelfPlayConfig,
    _assemble_dataset,
    load_selfplay_config,
    run_self_play,
)
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
    metrics = select_graphscript_tasks(
        [selected_task, one_hop], output, backend=toy_graph()
    )
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
    assert config.graph_snapshot == "kqapro-v1"
    assert config.interaction_mode == "tool"
    assert config.program_profile == "full"


def test_default_selfplay_file_uses_kqapro_snapshot() -> None:
    config_path = Path(__file__).parents[2] / "configs/training/selfplay.yaml"
    assert load_selfplay_config(config_path).graph_snapshot == "kqapro-v1"


def test_comparison_profile_requires_relation_catalog() -> None:
    with pytest.raises(ValueError, match="comparison profile requires relation_catalog"):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "val_data": "val.parquet",
                "questioner_seeds": "seeds.parquet",
                "interaction_mode": "tool",
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
    invalid = _ScriptedOpponent(
        tmp_path,
        [_tool_call("graph_search", {"entity_ids": ["cara"]}, 0)],
    )
    rejected = asyncio.run(
        invalid.rollout(
            _task(),
            toy_graph(),
            allowed_relations=("works_at", "located_in"),
            max_edge_visits=10,
        )
    )
    assert rejected["passed"] == 0.0
    assert rejected["edge_visits"] == 0.0

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
