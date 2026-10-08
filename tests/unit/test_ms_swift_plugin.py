from __future__ import annotations

import asyncio
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


def test_reward_completion_normalizes_optional_leading_thinking(plugin: Any) -> None:
    payload = '{"version":"0.1","ops":[]}'

    assert plugin._reward_completion(f"<think>\n\n</think>\n\n{payload}") == payload
    assert plugin._reward_completion(f"<think>reason</think>{payload}") == payload
    assert plugin._reward_completion(payload) == payload


def test_reward_completion_preserves_questioner_envelope(plugin: Any) -> None:
    payload = '{"question":"Who?","program":{"version":"0.3","ops":[]}}'

    assert plugin._reward_completion(f"<think>reason</think>{payload}") == payload


@pytest.mark.parametrize(
    "response",
    [
        '</tool_call>{"version":"0.3","ops":[]}',
        'prefix {"version":"0.3","ops":[]} trailing',
        '```json\n{"version":"0.3","ops":[]}\n```',
        '</tool_call>{"question":"Who?","program":{"version":"0.3","ops":[]}}',
    ],
)
def test_reward_completion_does_not_repair_actor_wrappers(
    plugin: Any, response: str
) -> None:
    assert plugin._reward_completion(response) == response


def test_dataset_registration_does_not_require_validation_data(
    plugin: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.parquet"
    train.touch()
    registered: list[object] = []
    monkeypatch.setattr(plugin, "register_dataset", registered.append)
    monkeypatch.setenv("GRAPHTASK_MS_SWIFT_DATA_KIND", "sft")
    monkeypatch.setenv("GRAPHTASK_MS_SWIFT_TRAIN_DATA", str(train))
    monkeypatch.delenv("GRAPHTASK_MS_SWIFT_VAL_DATA", raising=False)

    plugin._register_data()

    assert [item.dataset_name for item in registered] == ["graphtask-train"]


def test_distributed_cleanup_destroys_initialized_process_group(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    destroyed: list[bool] = []
    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: True
    distributed.destroy_process_group = lambda: destroyed.append(True)
    torch = types.ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)

    plugin._destroy_distributed_process_group()

    assert destroyed == [True]


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


def test_evidence_scheduler_continues_after_an_early_answer(plugin: Any) -> None:
    scheduler = plugin.GraphTaskCurriculumSolverScheduler(max_turns=4)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "evidence_solver",
                "graph_snapshot": "kilt-2019-08-01-v1",
                "topic_entity_ids": [],
            }
        },
    )

    assert scheduler.check_finished(request, _choice(None), 1) is False
    result = scheduler.step(request, _choice(None), 1)

    assert result["rollout_infos"]["early_answer_attempts"] == 1
    assert request.messages[-1] == {
        "role": "user",
        "content": "Evidence protocol incomplete. Call text_search before answering.",
    }
    state = request.data_dict["_graphtask_session"]
    state["selected_passage_keys"] = ["1:0"]
    assert scheduler.check_finished(request, _choice(None), 2) is False
    state["evidence_actions"] = [
        {"op": "retrieve"},
        {"op": "expand"},
        {"op": "select_evidence", "passage_keys": ["1:0"]},
    ]
    assert scheduler.check_finished(request, _choice(None), 2) is True


@pytest.mark.parametrize("reward_variant", ["ecp_v4", "ecp_v5"])
def test_causal_ecp_scheduler_requires_two_stage_selection(
    plugin: Any, reward_variant: str
) -> None:
    scheduler = plugin.GraphTaskCurriculumSolverScheduler(max_turns=4)
    request = SimpleNamespace(
        messages=[],
        data_dict={
            "extra_info": {
                "role": "evidence_solver",
                "graph_snapshot": "kilt-2019-08-01-v1",
                "evidence_reward_variant": reward_variant,
            },
            "_graphtask_session": {
                "calls": 3,
                "valid_calls": 3,
                "invalid_calls": 0,
                "visible_entities": [],
                "observed_passages": [],
                "selected_passage_keys": ["1:0", "2:0"],
                "evidence_actions": [
                    {
                        "op": "retrieve",
                        "passages": [{"passage_key": "1:0"}],
                    },
                    {
                        "op": "select_evidence",
                        "passage_keys": ["1:0", "2:0"],
                    },
                    {
                        "op": "expand",
                        "passages": [{"passage_key": "2:0"}],
                    },
                ],
            },
        },
    )

    assert scheduler.check_finished(request, _choice(None), 3) is False
    state = request.data_dict["_graphtask_session"]
    state["evidence_actions"].append(
        {"op": "select_evidence", "passage_keys": ["1:0", "2:0"]}
    )
    assert scheduler.check_finished(request, _choice(None), 3) is True


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
    assert payload[0]["passage_key"] == "123:0"
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


def test_ms_swift_reward_keeps_training_when_opponent_is_temporarily_unavailable(
    plugin: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from graphtask_r1.training.opponent import OpponentUnavailable

    async def unavailable(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        raise OpponentUnavailable("temporary text/plain gateway response")

    caplog.set_level("INFO", logger="graphtask_r1.training.ms_swift_plugin")
    monkeypatch.setattr(plugin, "compute_score", unavailable)
    reward = plugin.GraphTaskReward()

    values = reward(
        ["candidate"],
        data_source=["graphtask/questioner"],
        ground_truth=["{}"],
        extra_info=[{"graph_snapshot": "toy-v1", "task_id": "task-1"}],
    )

    assert values == [0.0]
    event = json.loads(caplog.records[-1].message)
    sample = event["sample_components"][0]
    assert sample["reason_codes"] == ["OPPONENT_UNAVAILABLE"]
    assert sample["components"]["opponent_unavailable"] == 1.0
    assert sample["components"]["opponent_required"] == 1.0
    assert sample["components"]["opponent_evaluated"] == 0.0
    assert sample["components"]["opponent_timeout"] == 0.0
    assert sample["components"]["reward_fallback"] == 1.0
    assert sample["components"]["reward_group_neutralized"] == 0.0


def test_ms_swift_reward_final_safety_net_preserves_timeout_reason(
    plugin: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from graphtask_r1.training.opponent import OpponentTimeout

    async def unavailable(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        raise OpponentTimeout(
            "opponent request deadline exceeded",
            stage="request",
            attempts=1,
            trace_id="task-timeout",
        )

    caplog.set_level("INFO", logger="graphtask_r1.training.ms_swift_plugin")
    monkeypatch.setattr(plugin, "compute_score", unavailable)
    reward = plugin.GraphTaskReward()

    assert reward(
        ["candidate"],
        data_source=["graphtask/questioner"],
        ground_truth=["{}"],
        extra_info=[
            {
                "graph_snapshot": "toy-v1",
                "task_id": "task-timeout",
                "questioner_reward_variant": "rule_program_question_v1",
            }
        ],
    ) == [0.0]

    event = json.loads(caplog.records[-1].message)
    sample = event["sample_components"][0]
    assert sample["reason_codes"] == ["OPPONENT_TIMEOUT"]
    assert sample["components"]["opponent_timeout"] == 1.0
    assert sample["components"]["opponent_unavailable"] == 1.0
    assert sample["components"]["reward_group_neutralized"] == 1.0


def test_ms_swift_reward_neutralizes_only_failed_prompt_group(
    plugin: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, object],
        *,
        backend: object | None = None,
    ) -> dict[str, float]:
        del data_source, ground_truth, extra_info, backend
        if solution_str == "timed-out":
            return {
                "score": 0.2,
                "raw_score": 0.4,
                "opponent_timeout": 1.0,
                "opponent_unavailable": 1.0,
                "reject_opponent_timeout": 1.0,
            }
        score = 0.6 if solution_str == "same-group" else 0.9
        return {"score": score, "raw_score": score}

    caplog.set_level("INFO", logger="graphtask_r1.training.ms_swift_plugin")
    monkeypatch.setattr(plugin, "compute_score", fake_compute_score)
    reward = plugin.GraphTaskReward()
    common_info = {
        "graph_snapshot": "toy-v1",
        "task_id": "shared-task",
        "questioner_reward_variant": "rule_program_question_v1",
    }

    values = reward(
        ["same-group", "timed-out", "healthy-group"],
        data_source=["graphtask/questioner"] * 3,
        ground_truth=["{}"] * 3,
        extra_info=[common_info, common_info, common_info],
        prompt_id=["prompt-a", "prompt-a", "prompt-b"],
    )

    assert values == [0.0, 0.0, 0.9]
    event = json.loads(caplog.records[-1].message)
    first, second, third = event["sample_components"]
    assert first["components"]["score_before_neutralization"] == 0.6
    assert first["components"]["raw_score"] == 0.0
    assert first["components"]["reward_group_neutralized"] == 1.0
    assert first["components"]["reward_fallback"] == 1.0
    assert second["components"]["score_before_neutralization"] == 0.2
    assert second["reason_codes"] == ["OPPONENT_TIMEOUT"]
    assert "reward_group_neutralized" not in third["components"]
    assert third["components"]["score"] == 0.9


def test_ms_swift_reward_falls_back_to_task_id_for_group_neutralization(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, object],
        *,
        backend: object | None = None,
    ) -> dict[str, float]:
        del data_source, ground_truth, extra_info, backend
        return {
            "score": 0.5,
            "raw_score": 0.5,
            "opponent_unavailable": float(solution_str == "unavailable"),
        }

    monkeypatch.setattr(plugin, "compute_score", fake_compute_score)
    reward = plugin.GraphTaskReward()

    values = reward(
        ["healthy-sibling", "unavailable", "other-task"],
        data_source=["graphtask/questioner"] * 3,
        ground_truth=["{}"] * 3,
        extra_info=[
            {
                "graph_snapshot": "toy-v1",
                "task_id": "task-a",
                "questioner_reward_variant": "rule_program_question_v1",
            },
            {
                "graph_snapshot": "toy-v1",
                "task_id": "task-a",
                "questioner_reward_variant": "rule_program_question_v1",
            },
            {
                "graph_snapshot": "toy-v1",
                "task_id": "task-b",
                "questioner_reward_variant": "rule_program_question_v1",
            },
        ],
    )

    assert values == [0.0, 0.0, 0.5]


def test_ms_swift_reward_merges_unavailable_groups_across_ranks(
    plugin: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_compute_score(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        return {"score": 0.75, "raw_score": 0.75}

    gathered_payloads: list[dict[str, object]] = []
    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: True
    distributed.get_world_size = lambda: 2

    def all_gather_object(output: list[object], payload: dict[str, object]) -> None:
        gathered_payloads.append(payload)
        output[:] = [
            payload,
            {"error": None, "unavailable_groups": ["remote-timeout"]},
        ]

    distributed.all_gather_object = all_gather_object
    torch = types.ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setenv("GRAPHTASK_RULE_GROUP_NEUTRALIZATION", "true")
    monkeypatch.setattr(plugin, "compute_score", fake_compute_score)
    caplog.set_level("INFO", logger="graphtask_r1.training.ms_swift_plugin")
    reward = plugin.GraphTaskReward()

    values = reward(
        ["remote sibling", "healthy"],
        data_source=["graphtask/questioner"] * 2,
        ground_truth=["{}", "{}"],
        extra_info=[
            {
                "graph_snapshot": "toy-v1",
                "task_id": "one",
                "questioner_reward_variant": "rule_program_question_v1",
            },
            {
                "graph_snapshot": "toy-v1",
                "task_id": "two",
                "questioner_reward_variant": "rule_program_question_v1",
            },
        ],
        prompt_id=["remote-timeout", "healthy"],
    )

    assert gathered_payloads == [{"error": None, "unavailable_groups": []}]
    assert values == [0.0, 0.75]
    event = json.loads(caplog.records[-1].message)
    remote_sibling = event["sample_components"][0]["components"]
    assert remote_sibling["score_before_neutralization"] == 0.75
    assert remote_sibling["reward_group_neutralized"] == 1.0
    assert remote_sibling["reward_fallback"] == 1.0


def test_ms_swift_reward_propagates_remote_rank_error_without_entering_group_collective(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_compute_score(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        return {"score": 0.75, "raw_score": 0.75}

    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: True
    distributed.get_world_size = lambda: 2

    def all_gather_object(output: list[object], payload: dict[str, object]) -> None:
        output[:] = [
            payload,
            {
                "error": {
                    "rank": "1",
                    "type": "ValueError",
                    "message": "bad reward config",
                },
                "unavailable_groups": [],
            },
        ]

    distributed.all_gather_object = all_gather_object
    torch = types.ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setenv("GRAPHTASK_RULE_GROUP_NEUTRALIZATION", "true")
    monkeypatch.setattr(plugin, "compute_score", fake_compute_score)
    reward = plugin.GraphTaskReward()

    with pytest.raises(
        RuntimeError,
        match="reward scoring failed on rank 1: ValueError: bad reward config",
    ):
        reward(
            ["candidate"],
            data_source=["graphtask/questioner"],
            ground_truth=["{}"],
            extra_info=[
                {
                    "graph_snapshot": "toy-v1",
                    "questioner_reward_variant": "rule_program_question_v1",
                }
            ],
            prompt_id=["prompt-a"],
        )


def test_ms_swift_reward_syncs_rule_preflight_error_across_ranks(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gathered_payloads: list[dict[str, object]] = []
    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: True
    distributed.get_world_size = lambda: 2

    def all_gather_object(output: list[object], payload: dict[str, object]) -> None:
        gathered_payloads.append(payload)
        output[:] = [payload, payload]

    distributed.all_gather_object = all_gather_object
    torch = types.ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setenv("GRAPHTASK_RULE_GROUP_NEUTRALIZATION", "true")
    reward = plugin.GraphTaskReward()

    with pytest.raises(ValueError, match="reward column has 2 rows; expected 1"):
        reward(
            ["candidate"],
            data_source=["graphtask/questioner", "extra"],
        )

    assert len(gathered_payloads) == 1
    error = gathered_payloads[0]["error"]
    assert isinstance(error, dict)
    assert error["type"] == "ValueError"


def test_ms_swift_reward_does_not_add_collective_to_non_rule_batches(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_compute_score(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        return {"score": 0.5, "raw_score": 0.5}

    collective_calls = 0
    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: True

    def all_gather_object(*_: object) -> None:
        nonlocal collective_calls
        collective_calls += 1

    distributed.all_gather_object = all_gather_object
    torch = types.ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setenv("GRAPHTASK_RULE_GROUP_NEUTRALIZATION", "false")
    monkeypatch.setattr(plugin, "compute_score", fake_compute_score)

    assert plugin.GraphTaskReward()(
        ["solver"],
        data_source=["graphtask/solver"],
        ground_truth=["{}"],
        extra_info=[{"graph_snapshot": "toy-v1"}],
    ) == [0.5]
    assert collective_calls == 0


def test_ms_swift_reward_limits_score_concurrency_per_rank(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak_active = 0

    async def fake_compute_score(*args: object, **kwargs: object) -> dict[str, float]:
        nonlocal active, peak_active
        del args, kwargs
        active += 1
        peak_active = max(peak_active, active)
        try:
            await asyncio.sleep(0.02)
        finally:
            active -= 1
        return {"score": 0.25, "raw_score": 0.25}

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(plugin, "compute_score", fake_compute_score)
    reward = plugin.GraphTaskReward()
    info = {
        "graph_snapshot": "toy-v1",
        "questioner_reward_variant": "rule_program_question_v1",
        "opponent_max_concurrency": 8,
        "opponent_samples": 2,
        "opponent_request_timeout_s": 1.0,
    }

    values = reward(
        [str(index) for index in range(6)],
        data_source=["graphtask/questioner"] * 6,
        ground_truth=["{}"] * 6,
        extra_info=[info] * 6,
        prompt_id=[f"prompt-{index}" for index in range(6)],
    )

    assert values == [0.25] * 6
    assert peak_active == 2


def test_ms_swift_reward_queue_wait_uses_request_deadline(
    plugin: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[str] = []

    async def slow_compute_score(
        data_source: str,
        solution_str: str,
        *args: object,
        **kwargs: object,
    ) -> dict[str, float]:
        del data_source, args, kwargs
        started.append(solution_str)
        await asyncio.sleep(1.0)
        return {"score": 0.5, "raw_score": 0.5}

    caplog.set_level("INFO", logger="graphtask_r1.training.ms_swift_plugin")
    monkeypatch.setattr(plugin, "compute_score", slow_compute_score)
    reward = plugin.GraphTaskReward()
    info = {
        "graph_snapshot": "toy-v1",
        "questioner_reward_variant": "rule_program_question_v1",
        "opponent_max_concurrency": 1,
        "opponent_samples": 1,
        "opponent_request_timeout_s": 0.02,
    }

    values = reward(
        ["first", "queued"],
        data_source=["graphtask/questioner"] * 2,
        ground_truth=["{}", "{}"],
        extra_info=[info, info],
        prompt_id=["first", "queued"],
    )

    assert values == [0.0, 0.0]
    assert started == ["first"]
    event = json.loads(caplog.records[-1].message)
    queued = event["sample_components"][1]["components"]
    assert queued["opponent_timeout_stage_reward_queue"] == 1.0
    assert queued["opponent_timeout"] == 1.0


def test_ms_swift_reward_passes_queue_adjusted_timeout_to_rule_scorer(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeouts: list[float] = []
    observed_deadlines: list[float] = []

    async def capture_timeout(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, object],
        *,
        backend: object | None = None,
    ) -> dict[str, float]:
        del data_source, solution_str, ground_truth, backend
        observed_timeouts.append(float(extra_info["opponent_request_timeout_s"]))
        observed_deadlines.append(
            float(extra_info["_opponent_reward_deadline_monotonic_s"])
        )
        await asyncio.sleep(0.03)
        return {"score": 0.5, "raw_score": 0.5}

    monkeypatch.setattr(plugin, "compute_score", capture_timeout)
    reward = plugin.GraphTaskReward()
    info = {
        "graph_snapshot": "toy-v1",
        "questioner_reward_variant": "rule_program_question_v1",
        "opponent_max_concurrency": 1,
        "opponent_samples": 1,
        "opponent_request_timeout_s": 0.2,
    }

    assert reward(
        ["first", "second"],
        data_source=["graphtask/questioner"] * 2,
        ground_truth=["{}", "{}"],
        extra_info=[info, info],
        prompt_id=["first", "second"],
    ) == [0.5, 0.5]

    assert len(observed_timeouts) == 2
    assert 0.0 < observed_timeouts[1] < observed_timeouts[0] <= 0.2
    assert len(observed_deadlines) == 2
    assert observed_deadlines[0] == pytest.approx(observed_deadlines[1], abs=0.01)


def test_ms_swift_reward_does_not_misclassify_application_timeout_error(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        raise TimeoutError("local scorer bug")

    monkeypatch.setattr(plugin, "compute_score", broken)
    reward = plugin.GraphTaskReward()

    with pytest.raises(TimeoutError, match="local scorer bug"):
        reward(
            ["candidate"],
            data_source=["graphtask/solver"],
            ground_truth=["{}"],
            extra_info=[{"graph_snapshot": "toy-v1"}],
        )


def test_ms_swift_reward_rejects_non_finite_components(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def non_finite(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        return {"score": float("nan"), "raw_score": float("nan")}

    monkeypatch.setattr(plugin, "compute_score", non_finite)
    reward = plugin.GraphTaskReward()

    with pytest.raises(ValueError, match="reward components must be finite"):
        reward(
            ["candidate"],
            data_source=["graphtask/solver"],
            ground_truth=["{}"],
            extra_info=[{"graph_snapshot": "toy-v1"}],
        )


def test_ms_swift_reward_does_not_swallow_external_cancellation(
    plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancelled(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin, "compute_score", cancelled)
    reward = plugin.GraphTaskReward()

    with pytest.raises(asyncio.CancelledError):
        reward(
            ["candidate"],
            data_source=["graphtask/questioner"],
            ground_truth=["{}"],
            extra_info=[{"graph_snapshot": "toy-v1"}],
        )


def test_curriculum_solver_reward_receives_scheduler_rollout_infos(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, object]] = []

    async def fake_compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, object],
        *,
        backend: object | None = None,
    ) -> dict[str, float]:
        del data_source, solution_str, ground_truth, backend
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


def test_ms_swift_reward_reuses_backend_per_snapshot(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[tuple[str, object]] = []
    received: list[object] = []

    def fake_backend_from_snapshot(snapshot: str) -> object:
        backend = object()
        created.append((snapshot, backend))
        return backend

    async def fake_compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, object],
        *,
        backend: object | None = None,
    ) -> dict[str, float]:
        del data_source, solution_str, ground_truth, extra_info
        assert backend is not None
        received.append(backend)
        return {"score": 0.25, "raw_score": 0.25}

    monkeypatch.setattr(plugin, "backend_from_snapshot", fake_backend_from_snapshot)
    monkeypatch.setattr(plugin, "compute_score", fake_compute_score)
    reward = plugin.GraphTaskReward()
    kwargs = {
        "data_source": ["graphtask/questioner", "graphtask/questioner"],
        "ground_truth": ["{}", "{}"],
        "extra_info": [
            {"graph_snapshot": "kqapro-v1"},
            {"graph_snapshot": "kqapro-v1"},
        ],
    }

    assert reward(["one", "two"], **kwargs) == [0.25, 0.25]
    assert reward(["three", "four"], **kwargs) == [0.25, 0.25]
    assert len(created) == 1
    assert len(received) == 4
    assert all(backend is received[0] for backend in received)


def test_grpo_launcher_can_select_curriculum_scheduler() -> None:
    project_root = Path(__file__).parents[2]
    launcher = (project_root / "scripts/train_ms_swift_grpo.sh").read_text()

    assert 'MULTI_TURN_SCHEDULER="${MULTI_TURN_SCHEDULER:-graphtask_solver}"' in launcher
    assert '--multi_turn_scheduler "$MULTI_TURN_SCHEDULER"' in launcher
    assert '--response_prefix "$RESPONSE_PREFIX"' in launcher
    assert '--template "${TEMPLATE:-qwen3}"' in launcher
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
