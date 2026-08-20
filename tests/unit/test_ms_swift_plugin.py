from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from graphtask_r1.schema import AnswerSet


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> Any:
    class FakePreprocessor:
        pass

    @dataclass
    class FakeDatasetMeta:
        dataset_name: str
        dataset_path: str
        preprocess_func: object

    class FakeORM:
        pass

    class FakeScheduler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            self.max_turns = kwargs.get("max_turns")

        def check_finished(
            self, infer_request: object, response_choice: object, current_turn: int
        ) -> bool:
            del infer_request, response_choice
            return bool(self.max_turns and current_turn >= self.max_turns)

    swift = types.ModuleType("swift")
    llm = types.ModuleType("swift.llm")
    dataset = types.ModuleType("swift.llm.dataset")
    dataset.DatasetMeta = FakeDatasetMeta
    dataset.RowPreprocessor = FakePreprocessor
    dataset.register_dataset = lambda value: value
    swift_plugin = types.ModuleType("swift.plugin")
    swift_plugin.ORM = FakeORM
    swift_plugin.orms = {}
    swift_plugin.multi_turns = {}
    multi_turn = types.ModuleType("swift.plugin.multi_turn")
    multi_turn.MultiTurnScheduler = FakeScheduler

    for name, module in {
        "swift": swift,
        "swift.llm": llm,
        "swift.llm.dataset": dataset,
        "swift.plugin": swift_plugin,
        "swift.plugin.multi_turn": multi_turn,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("GRAPHTASK_MS_SWIFT_DATA_KIND", raising=False)
    module_name = "graphtask_r1.training.ms_swift_plugin"
    sys.modules.pop(module_name, None)
    loaded = importlib.import_module(module_name)
    yield loaded
    sys.modules.pop(module_name, None)


def _choice_with_arguments(
    name: str | None, arguments: dict[str, object] | None = None
) -> SimpleNamespace:
    calls = []
    if name:
        calls.append(
            SimpleNamespace(
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(
                        arguments
                        or {
                            "entity_ids": ["alice"],
                            "direction": "out",
                            "relation_ids": ["works_at"],
                            "limit": 5,
                        }
                    ),
                )
            )
        )
    return SimpleNamespace(message=SimpleNamespace(tool_calls=calls))


def _choice(name: str | None) -> SimpleNamespace:
    return _choice_with_arguments(name)


def test_reward_completion_removes_only_prefilled_empty_thinking(plugin: Any) -> None:
    payload = '{"version":"0.1","ops":[]}'

    assert plugin._reward_completion(f"<think>\n\n</think>\n\n{payload}") == payload
    assert plugin._reward_completion(f"<think>reason</think>{payload}") != payload


def test_graphscript_mode_does_not_register_multi_turn_scheduler(plugin: Any) -> None:
    del plugin
    from swift.plugin import multi_turns

    assert "graphtask_solver" not in multi_turns


def test_tool_mode_registers_multi_turn_scheduler(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    del plugin
    monkeypatch.setenv("INTERACTION_MODE", "tool")
    module_name = "graphtask_r1.training.ms_swift_plugin"
    sys.modules.pop(module_name, None)

    importlib.import_module(module_name)
    from swift.plugin import multi_turns
    from swift.plugin.multi_turn import MultiTurnScheduler

    assert multi_turns["graphtask_solver"].__name__ == "GraphTaskSolverScheduler"
    assert (
        multi_turns["graphtask_curriculum_solver"].__name__ == "GraphTaskCurriculumSolverScheduler"
    )
    assert issubclass(
        multi_turns["graphtask_curriculum_solver"], MultiTurnScheduler
    )


def test_solver_scheduler_keeps_json_session_state_per_request(plugin: Any) -> None:
    scheduler = plugin.GraphTaskSolverScheduler(max_turns=8)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "solver",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
                "task_id": "task-1",
                "max_edge_visits": 10,
            }
        },
    )

    result = scheduler.step(request, _choice("graph_search"), 1)

    assert result is request
    assert request.messages[-1]["role"] == "tool"
    assert json.loads(request.messages[-1]["content"]) == [
        {"subject": "alice", "relation": "works_at", "object": "acme"}
    ]
    json.dumps(request.data_dict)
    assert scheduler.check_finished(request, _choice("graph_search"), 1) is False
    assert scheduler.check_finished(request, _choice(None), 2) is True


def test_solver_scheduler_returns_structured_invalid_call(plugin: Any) -> None:
    scheduler = plugin.GraphTaskSolverScheduler(max_turns=8)
    request = SimpleNamespace(
        messages=[],
        data_dict={"extra_info": {"role": "solver", "graph_snapshot": "toy-v1"}},
    )

    result = scheduler.step(request, _choice("unknown_tool"), 1)

    assert result is request
    assert request.data_dict["_graphtask_session"]["invalid_calls"] == 1
    error = json.loads(request.messages[-1]["content"])["error"]
    assert error["reason_code"] == "INVALID_TOOL_CALL"


def test_curriculum_solver_scheduler_returns_cumulative_rollout_infos(plugin: Any) -> None:
    scheduler = plugin.GraphTaskCurriculumSolverScheduler(max_turns=8)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "solver",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
                "task_id": "task-curriculum",
                "max_edge_visits": 10,
            }
        },
    )

    result = scheduler.step(request, _choice("graph_search"), 1)

    assert result["infer_request"] is request
    assert result["rollout_infos"] == {
        "calls": 1,
        "valid_calls": 1,
        "invalid_calls": 0,
        "edge_visits": 1,
        "new_visible_entities": 1,
    }
    json.dumps(result["rollout_infos"])


def test_curriculum_solver_scheduler_counts_malformed_calls(plugin: Any) -> None:
    scheduler = plugin.GraphTaskCurriculumSolverScheduler(max_turns=8)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "solver",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
            }
        },
    )
    malformed = SimpleNamespace(
        message=SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(name="graph_search", arguments="{not-json")
                )
            ]
        )
    )

    result = scheduler.step(request, malformed, 1)

    assert result["rollout_infos"] == {
        "calls": 1,
        "valid_calls": 0,
        "invalid_calls": 1,
        "edge_visits": 0,
        "new_visible_entities": 0,
    }
    error = json.loads(request.messages[-1]["content"])["error"]
    assert error["reason_code"] == "INVALID_TOOL_CALL"


def test_curriculum_scheduler_allows_questioner_graph_search(plugin: Any) -> None:
    scheduler = plugin.GraphTaskCurriculumSolverScheduler(max_turns=8)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "questioner",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
                "max_edge_visits": 10,
            }
        },
    )

    result = scheduler.step(request, _choice("graph_search"), 1)

    assert result["rollout_infos"]["valid_calls"] == 1
    assert json.loads(request.messages[-1]["content"]) == [
        {"subject": "alice", "relation": "works_at", "object": "acme"}
    ]


def test_curriculum_scheduler_executes_questioner_program(plugin: Any) -> None:
    scheduler = plugin.GraphTaskCurriculumSolverScheduler(max_turns=8)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "questioner",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
            }
        },
    )
    program = {
        "op": "hop",
        "input": {"op": "entity", "entity_id": "alice"},
        "relation": "works_at",
        "direction": "out",
    }

    result = scheduler.step(
        request,
        _choice_with_arguments("execute_program", {"program": program}),
        1,
    )

    assert result["rollout_infos"]["valid_calls"] == 1
    assert AnswerSet.model_validate_json(request.messages[-1]["content"]) == (
        AnswerSet.entities(["acme"])
    )


def test_curriculum_scheduler_rejects_solver_execute_program(plugin: Any) -> None:
    scheduler = plugin.GraphTaskCurriculumSolverScheduler(max_turns=8)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "solver",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
            }
        },
    )

    result = scheduler.step(
        request,
        _choice_with_arguments(
            "execute_program", {"program": {"op": "entity", "entity_id": "alice"}}
        ),
        1,
    )

    assert result["rollout_infos"]["valid_calls"] == 0
    assert result["rollout_infos"]["invalid_calls"] == 1
    error = json.loads(request.messages[-1]["content"])["error"]
    assert error["reason_code"] == "INVALID_TOOL_CALL"


def test_solver_scheduler_executes_compact_query(plugin: Any) -> None:
    scheduler = plugin.GraphTaskSolverScheduler(max_turns=8)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "solver",
                "graph_snapshot": "toy-v1",
                "task_id": "task-compact",
                "max_returned_entities": 10,
            }
        },
    )
    choice = _choice_with_arguments(
        "graph_search",
        {
            "query": {
                "root": {"kind": "all_entities"},
                "steps": [{"op": "filter_type", "type_ids": ["person"]}],
                "return_count": True,
            }
        },
    )

    scheduler.step(request, choice, 1)

    payload = json.loads(request.messages[-1]["content"])
    assert payload["count"] == 3
    assert payload["truncated"] is False


def test_solver_scheduler_executes_bounded_text_search(plugin: Any) -> None:
    class SearchBackend:
        def search_text(
            self,
            query: str,
            *,
            limit: int,
            max_chars: int,
            trace_id: str | None,
        ) -> list[dict[str, object]]:
            assert query == "Caledonian Brewery"
            assert limit == 2
            assert max_chars == 1000
            assert trace_id == "openqa-1:1"
            return [
                {
                    "page_id": "123",
                    "paragraph_id": 0,
                    "title": "Caledonian Brewery",
                    "text": "The brewery is in Edinburgh.",
                    "score": -1.0,
                }
            ]

    scheduler = plugin.GraphTaskSolverScheduler(max_turns=8)
    scheduler._backends["kilt-2019-08-01-v1"] = SearchBackend()
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "solver",
                "graph_snapshot": "kilt-2019-08-01-v1",
                "task_id": "openqa-1",
                "topic_entity_ids": [],
                "text_search_enabled": True,
                "max_text_search_results": 3,
                "max_passage_chars": 1000,
            }
        },
    )

    scheduler.step(
        request,
        _choice_with_arguments("text_search", {"query": "Caledonian Brewery", "limit": 2}),
        1,
    )

    payload = json.loads(request.messages[-1]["content"])
    assert payload[0]["page_id"] == "123"
    assert request.data_dict["_graphtask_session"]["visible_entities"] == ["123"]


def test_ms_swift_reward_reuses_existing_gold_and_logs_components(
    plugin: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    caplog.set_level("INFO", logger="graphtask_r1.training.ms_swift_plugin")
    metrics_dir = tmp_path / "reward_metrics"
    monkeypatch.setenv("GRAPHTASK_REWARD_METRICS_DIR", str(metrics_dir))
    monkeypatch.setenv("RL_ALGORITHM", "reinforce_plus_plus")
    monkeypatch.setenv("RANK", "2")
    reward = plugin.GraphTaskReward()

    values = reward(
        ['<answer>["acme"]</answer>'],
        data_source=["graphtask/solver"],
        ground_truth=[AnswerSet.entities(["acme"]).model_dump_json()],
        extra_info=[
            {
                "graph_snapshot": "toy-v1",
                "interaction_mode": "tool",
                "role_weight": 1.0,
            }
        ],
    )

    assert values == [1.0]
    event = json.loads(caplog.records[-1].message)
    assert event["event"] == "graphtask_reward_components"
    assert event["rl_algorithm"] == "reinforce_plus_plus"
    assert event["means"]["f1"] == 1.0
    assert event["means"]["exact_match"] == 1.0
    assert event["roles"]["solver"]["means"]["unweighted_score"] == 1.0
    assert event["sample_components"] == [
        {
            "batch_index": 0,
            "role": "solver",
            "task_id": "",
            "reason_codes": [],
                "components": {
                    "answer_f1": 1.0,
                    "exact_match": 1.0,
                    "exact_output": 1.0,
                    "executable": 1.0,
                    "f1": 1.0,
                    "format": 1.0,
                    "json_valid": 1.0,
                    "precision": 1.0,
                    "raw_score": 1.0,
                    "recall": 1.0,
                    "reward_stage": 6.0,
                    "schema_valid": 1.0,
                    "score": 1.0,
                    "structure_valid": 1.0,
                    "unweighted_score": 1.0,
                },
        }
    ]
    persisted = json.loads(
        (metrics_dir / "reward_components.rank-2.jsonl").read_text().strip()
    )
    assert persisted == event


def test_curriculum_solver_reward_receives_scheduler_rollout_infos(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, object]] = []

    async def fake_compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, object],
    ) -> dict[str, float]:
        del data_source, solution_str, ground_truth
        captured.append(extra_info)
        return {"score": 0.25, "raw_score": 0.25}

    monkeypatch.setattr(plugin, "compute_score", fake_compute_score)
    reward = plugin.GraphTaskReward()

    values = reward(
        ['<answer>["acme"]</answer>'],
        data_source=["graphtask/solver"],
        ground_truth=[AnswerSet.entities(["acme"]).model_dump_json()],
        extra_info=[
            {
                "graph_snapshot": "toy-v1",
                "interaction_mode": "tool",
                "solver_reward_variant": "curriculum_v3",
                "curriculum_phase": 1,
            }
        ],
        rollout_infos=[
            {
                "calls": 2,
                "valid_calls": 1,
                "invalid_calls": 1,
                "edge_visits": 3,
                "new_visible_entities": 2,
                "num_turns": 3,
            }
        ],
    )

    assert values == [0.25]
    assert captured[0]["curriculum_phase"] == 1
    assert captured[0]["solver_rollout"] == {
        "calls": 2,
        "valid_calls": 1,
        "invalid_calls": 1,
        "edge_visits": 3,
        "new_visible_entities": 2,
        "num_turns": 3,
    }

    reward(
        ['<answer>["acme"]</answer>'],
        data_source=["graphtask/solver"],
        ground_truth=[AnswerSet.entities(["acme"]).model_dump_json()],
        extra_info=[
            {
                "graph_snapshot": "toy-v1",
                "interaction_mode": "tool",
                "solver_reward_variant": "legacy",
            }
        ],
        rollout_infos=[{"calls": 9}],
    )
    assert "solver_rollout" not in captured[1]


def test_grpo_launcher_can_select_curriculum_scheduler() -> None:
    project_root = Path(__file__).parents[2]
    launcher = (project_root / "scripts/train_ms_swift_grpo.sh").read_text()

    assert 'MULTI_TURN_SCHEDULER="${MULTI_TURN_SCHEDULER:-graphtask_solver}"' in launcher
    assert '--multi_turn_scheduler "$MULTI_TURN_SCHEDULER"' in launcher
    assert '--response_prefix "$RESPONSE_PREFIX"' in launcher
    assert 'TRAIN_DATA="${TRAIN_DATA:-${SOLVER_RL_TRAIN_DATA:-}}"' in launcher


def test_ms_swift_reward_logs_questioner_stage_and_reason_per_sample(
    plugin: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    caplog.set_level("INFO", logger="graphtask_r1.training.ms_swift_plugin")
    monkeypatch.setenv("GRAPHTASK_REWARD_METRICS_DIR", str(tmp_path / "reward_metrics"))
    reward = plugin.GraphTaskReward()

    values = reward(
        ["not-json"],
        data_source=["graphtask/questioner"],
        ground_truth=["{}"],
        extra_info=[
            {
                "graph_snapshot": "toy-v1",
                "interaction_mode": "graphscript",
                "graphscript_version": "0.3",
                "role_weight": 0.35,
                "task_id": "questioner-1",
            }
        ],
    )

    assert values == [-0.35]
    event = json.loads(caplog.records[-1].message)
    sample = event["sample_components"][0]
    assert sample["task_id"] == "questioner-1"
    assert sample["reason_codes"] == ["NON_JSON"]
    assert sample["components"]["reward_stage"] == 0.0
    assert sample["components"]["raw_score"] == -1.0
