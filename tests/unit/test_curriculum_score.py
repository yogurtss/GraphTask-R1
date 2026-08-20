from __future__ import annotations

import asyncio
import json

import pytest

from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import AnswerSet, Entity, Hop, TaskProposal
from graphtask_r1.training import ms_swift_reward as reward_module
from graphtask_r1.training.ms_swift_reward import compute_score
from graphtask_r1.training.parsing import parse_questioner_graphscript_output
from graphtask_r1.training.prompts import role_prompt


def _script(first: str = "works_at", second: str = "located_in") -> dict[str, object]:
    return {
        "version": "0.1",
        "ops": [
            {"op": "start", "entity": "$seed", "out": "h0"},
            {
                "op": "follow",
                "in": "h0",
                "relation": first,
                "direction": "out",
                "limit": 100,
                "out": "h1",
            },
            {
                "op": "follow",
                "in": "h1",
                "relation": second,
                "direction": "out",
                "limit": 100,
                "out": "h2",
            },
            {"op": "require_unique", "in": "h2"},
            {"op": "emit", "in": "h2"},
        ],
    }


def _info(stage: str) -> dict[str, object]:
    return {
        "interaction_mode": "graphscript",
        "graphscript_version": "0.1",
        "graph_snapshot": "toy-v1",
        "topic_entity_ids": ["alice"],
        "allowed_relations": ["friend", "friend_of_friend", "works_at", "located_in"],
        "max_edge_visits": 10,
        "questioner_reward_variant": "curriculum_v3",
        "curriculum_phase": stage,
        "opponent_url": "http://unused",
        "opponent_samples": 2,
    }


def _wrapped(question: str, *, first: str = "works_at", second: str = "located_in") -> str:
    return json.dumps({"question": question, "program": _script(first, second)})


def _tool_task(question: str) -> str:
    payload = {
        "question": question,
        "topic_entities": ["alice"],
        "program": {
            "op": "hop",
            "input": {
                "op": "hop",
                "input": {"op": "entity", "entity_id": "alice"},
                "relation": "works_at",
                "direction": "out",
            },
            "relation": "located_in",
            "direction": "out",
        },
    }
    return f"<task>{json.dumps(payload)}</task>"


def test_question_program_contract_and_legacy_code_only_parser() -> None:
    question, wrapped = parse_questioner_graphscript_output(
        _wrapped("Where is Alice's workplace located?")
    )
    old_question, code_only = parse_questioner_graphscript_output(json.dumps(_script()))

    assert question == "Where is Alice's workplace located?"
    assert wrapped == code_only
    assert old_question is None
    prompt = role_prompt(
        "questioner",
        "seed=alice",
        interaction_mode="graphscript",
        graphscript_version="0.1",
        questioner_contract="question_program",
    )
    assert '"question":"...","program"' in prompt[0]["content"]
    assert "require_unique" in prompt[0]["content"]
    assert '"op":"start","entity":"$seed","out":"h0"' in prompt[0]["content"]
    tool_prompt = role_prompt(
        "questioner",
        "seed=alice",
        interaction_mode="tool",
        questioner_contract="question_program",
    )
    assert '<task>{"question":"...","topic_entities"' in tool_prompt[0]["content"]


def test_production_scores_question_and_code_without_calling_opponent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_opponent(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        raise AssertionError("production must not call the opponent")

    monkeypatch.setattr(reward_module, "request_opponent", forbidden_opponent)
    wrapped = asyncio.run(
        compute_score(
            "graphtask/questioner",
            _wrapped("Where is Alice's workplace located?"),
            "{}",
            _info("production"),
        )
    )
    code_only = asyncio.run(
        compute_score(
            "graphtask/questioner",
            json.dumps(_script()),
            "{}",
            _info("production"),
        )
    )
    truncated = asyncio.run(
        compute_score(
            "graphtask/questioner",
            (
                '{"question":"Where is Alice located?","program":'
                '{"version":"0.1","ops":[{"op":"start","entity":"$seed"}'
            ),
            "{}",
            _info("production"),
        )
    )

    assert wrapped["raw_score"] == pytest.approx(1.0)
    assert 0.0 < truncated["raw_score"] < code_only["raw_score"] < wrapped["raw_score"]
    assert truncated["milestone_question_present"] == 1.0
    assert truncated["milestone_valid_operation_fraction"] > 0.0
    assert code_only["milestone_code_present"] == 1.0
    assert code_only["milestone_question_present"] == 0.0
    assert code_only["reject_missing_question"] == 1.0


def test_grounding_uses_generated_question_and_relaxes_only_quality_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_opponent(*args: object, **kwargs: object) -> dict[str, float]:
        del args
        captured.update(kwargs)
        return {
            "pass_rate": 0.5,
            "program_parse_rate": 1.0,
            "execution_rate_given_parse": 1.0,
            "semantic_success_given_execution": 0.75,
            "novelty_structural": 1.0,
            "novelty_textual": 1.0,
        }

    monkeypatch.setattr(reward_module, "request_opponent", fake_opponent)
    question = "Who is reached by following friend twice from Alice?"
    score = asyncio.run(
        compute_score(
            "graphtask/questioner",
            _wrapped(question, first="friend", second="friend"),
            "{}",
            _info("grounding"),
        )
    )

    assert captured["generated_question"] == question
    assert captured["allowed_rejection_reasons"] == frozenset(
        {"SHORTCUT_FOUND", "SHORTCUT_UNKNOWN", "REDUNDANT_CONDITION"}
    )
    assert score["reject_shortcut_found"] == 1.0
    assert score["milestone_certified"] == 0.0
    assert 0.0 < score["milestone_question_program_alignment"] <= 1.0
    assert score["opponent_semantic_success_given_execution"] == 0.75


def test_grounding_answer_leak_keeps_dense_credit_without_opponent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_opponent(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        raise AssertionError("leaked questions must not reach the opponent")

    monkeypatch.setattr(reward_module, "request_opponent", forbidden_opponent)
    score = asyncio.run(
        compute_score(
            "graphtask/questioner",
            _wrapped("Is Paris the answer?"),
            "{}",
            _info("grounding"),
        )
    )

    assert score["reject_answer_leak"] == 1.0
    assert score["milestone_no_answer_leak"] == 0.0
    assert score["milestone_program_executable"] == 1.0
    assert score["raw_score"] > 0.5


def test_tool_grounding_uses_task_question_as_generated_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_opponent(*args: object, **kwargs: object) -> dict[str, float]:
        del args
        captured.update(kwargs)
        return {
            "pass_rate": 0.5,
            "program_parse_rate": 1.0,
            "execution_rate_given_parse": 1.0,
            "semantic_success_given_execution": 0.5,
            "novelty_structural": 1.0,
            "novelty_textual": 1.0,
        }

    monkeypatch.setattr(reward_module, "request_opponent", fake_opponent)
    question = "Where is the organization that Alice works at located?"
    info = _info("grounding")
    info["interaction_mode"] = "tool"
    score = asyncio.run(
        compute_score(
            "graphtask/questioner",
            _tool_task(question),
            "{}",
            info,
        )
    )

    assert captured["generated_question"] == question
    assert captured["interaction_mode"] == "tool"
    proposal = captured["proposal"]
    assert isinstance(proposal, TaskProposal)
    assert proposal.paraphrase == question
    assert score["milestone_question_present"] == 1.0
    assert score["milestone_program_executable"] == 1.0


def test_low_question_program_alignment_does_not_reach_opponent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_opponent(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        raise AssertionError("misaligned questions must not reach the opponent")

    monkeypatch.setattr(reward_module, "request_opponent", forbidden_opponent)
    score = asyncio.run(
        compute_score(
            "graphtask/questioner",
            _wrapped("What happened?"),
            "{}",
            _info("grounding"),
        )
    )

    assert score["milestone_question_program_alignment"] < 0.35
    assert score["opponent_parse_rate"] == 0.0


def test_curriculum_solver_keeps_dense_syntax_and_solve_ordering() -> None:
    info = {
        **_info("production"),
        "solver_reward_variant": "curriculum_v3",
    }
    non_json = asyncio.run(
        compute_score(
            "graphtask/solver",
            "not-json",
            AnswerSet.entities(["paris"]).model_dump_json(),
            info,
        )
    )
    partial_json = asyncio.run(
        compute_score(
            "graphtask/solver",
            json.dumps({"version": "0.1", "ops": [{"op": "start"}]}),
            AnswerSet.entities(["paris"]).model_dump_json(),
            info,
        )
    )
    truncated_json = asyncio.run(
        compute_score(
            "graphtask/solver",
            '{"version":"0.1","ops":[{"op":"start","entity":"$seed"}',
            AnswerSet.entities(["paris"]).model_dump_json(),
            info,
        )
    )
    valid = asyncio.run(
        compute_score(
            "graphtask/solver",
            json.dumps(_script()),
            AnswerSet.entities(["paris"]).model_dump_json(),
            info,
        )
    )

    assert (
        0.0
        < non_json["raw_score"]
        < truncated_json["raw_score"]
        < partial_json["raw_score"]
        < valid["raw_score"]
    )
    assert truncated_json["milestone_valid_prefix_fraction"] > 0.0
    assert non_json["stage_syntax"] == 1.0
    assert partial_json["reject_invalid_schema"] == 1.0

    solve_info = {**info, "curriculum_phase": "frontier"}
    wrong = asyncio.run(
        compute_score(
            "graphtask/solver",
            json.dumps(_script()),
            AnswerSet.entities(["bob"]).model_dump_json(),
            solve_info,
        )
    )
    exact = asyncio.run(
        compute_score(
            "graphtask/solver",
            json.dumps(_script()),
            AnswerSet.entities(["paris"]).model_dump_json(),
            solve_info,
        )
    )
    assert wrong["raw_score"] < exact["raw_score"] == pytest.approx(1.0)


def test_curriculum_tool_solver_uses_gold_literal_kind_and_process_signals() -> None:
    info = {
        **_info("frontier"),
        "interaction_mode": "tool",
        "solver_reward_variant": "curriculum_v3",
        "solver_rollout": {
            "calls": 2,
            "valid_calls": 2,
            "invalid_calls": 0,
            "edge_visits": 2,
            "new_visible_entities": 2,
        },
    }
    score = asyncio.run(
        compute_score(
            "graphtask/solver",
            '<answer>["Paris"]</answer>',
            AnswerSet.literals(["Paris"]).model_dump_json(),
            info,
        )
    )

    assert score["exact_match"] == 1.0
    assert score["milestone_valid_tool_call_fraction"] == 1.0
    assert score["milestone_answer_parse_valid"] == 1.0


def test_relaxed_certificate_preserves_generated_question_and_executed_gold() -> None:
    graph = toy_graph()
    program = Hop(
        input=Hop(input=Entity(entity_id="alice"), relation="friend"),
        relation="friend",
    )
    question = "Who is reached by following friend twice from Alice?"
    task = certify_proposal(
        TaskProposal(topic_entities=("alice",), program=program, paraphrase=question),
        graph,
        graph_snapshot="toy-v1",
        generated_question=question,
        allowed_rejection_reasons=frozenset({"SHORTCUT_FOUND"}),
    )

    assert task.question == question
    assert task.gold_answers == graph.execute_program(program)
    assert task.verification.shortcut_found is True
