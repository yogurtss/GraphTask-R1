from __future__ import annotations

from dataclasses import replace

import pytest

from graphtask_r1.rewards import (
    OpponentSignals,
    QuestionerMilestones,
    SolverMilestones,
    questioner_curriculum_reward,
    solver_curriculum_reward,
)


def _grounded_questioner() -> QuestionerMilestones:
    return QuestionerMilestones(
        question_present=1.0,
        code_present=1.0,
        json_valid=1.0,
        schema_fraction=1.0,
        valid_operation_fraction=1.0,
        valid_prefix_fraction=1.0,
        seed_coverage=1.0,
        relation_valid_fraction=1.0,
        handle_valid_fraction=1.0,
        type_valid_fraction=1.0,
        executable_prefix_fraction=1.0,
        program_executable=1.0,
        answer_nonempty=1.0,
        cardinality_valid=1.0,
        question_program_alignment=1.0,
        no_answer_leak=1.0,
        certified=1.0,
    )


def test_questioner_production_reward_is_dense_and_auditable() -> None:
    partial = questioner_curriculum_reward(
        QuestionerMilestones(
            question_present=1.0,
            code_present=1.0,
            json_valid=1.0,
            schema_fraction=0.5,
            valid_operation_fraction=0.25,
            valid_prefix_fraction=0.4,
        ),
        stage="production",
    )
    improved = questioner_curriculum_reward(
        QuestionerMilestones(
            question_present=1.0,
            code_present=1.0,
            json_valid=1.0,
            schema_fraction=0.75,
            valid_operation_fraction=0.5,
            valid_prefix_fraction=0.6,
        ),
        stage="production",
    )

    assert 0.0 < partial.total < improved.total < 1.0
    assert partial.components["milestone_schema_fraction"] == 0.5
    assert partial.components["production_score"] == partial.total
    assert partial.components["stage_production"] == 1.0
    assert partial.metadata["stage"] == "production"


def test_questioner_grounding_does_not_require_certification_for_credit() -> None:
    milestones = _grounded_questioner()
    uncertified = QuestionerMilestones(
        **{**milestones.__dict__, "certified": 0.0}
    )
    reward = questioner_curriculum_reward(uncertified, stage="grounding")

    assert reward.total > 0.9
    assert reward.components["milestone_certified"] == 0.0
    assert reward.components["grounding_contribution"] > 0.0
    assert reward.components["curriculum_total"] == reward.total


def test_questioner_frontier_ignores_semantic_outcome_until_interface_is_ready() -> None:
    milestones = _grounded_questioner()
    failed_semantics = questioner_curriculum_reward(
        milestones,
        stage="frontier",
        opponent=OpponentSignals(
            parse_rate=0.0,
            execution_rate_given_parse=0.0,
            semantic_success_given_execution=0.0,
        ),
    )
    perfect_semantics = questioner_curriculum_reward(
        milestones,
        stage="frontier",
        opponent=OpponentSignals(
            parse_rate=0.0,
            execution_rate_given_parse=0.0,
            semantic_success_given_execution=1.0,
        ),
    )

    assert failed_semantics.total == perfect_semantics.total == 1.0
    assert failed_semantics.components["frontier_weight_effective"] == 0.0
    assert failed_semantics.components["opponent_interface_readiness"] == 0.0


def test_questioner_frontier_uses_conditional_semantics_when_solver_is_ready() -> None:
    milestones = _grounded_questioner()
    at_frontier = questioner_curriculum_reward(
        milestones,
        stage="frontier",
        opponent=OpponentSignals(
            parse_rate=1.0,
            execution_rate_given_parse=1.0,
            semantic_success_given_execution=0.5,
        ),
    )
    too_hard = questioner_curriculum_reward(
        milestones,
        stage="frontier",
        opponent=OpponentSignals(
            parse_rate=1.0,
            execution_rate_given_parse=1.0,
            semantic_success_given_execution=0.0,
        ),
    )
    partially_ready = questioner_curriculum_reward(
        milestones,
        stage="frontier",
        opponent=OpponentSignals(
            parse_rate=0.5,
            execution_rate_given_parse=0.5,
            semantic_success_given_execution=0.0,
        ),
    )

    assert at_frontier.total == pytest.approx(1.0)
    assert at_frontier.total > partially_ready.total > too_hard.total
    assert partially_ready.components["opponent_interface_readiness"] == 0.25
    assert partially_ready.components["frontier_weight_effective"] == pytest.approx(0.0875)
    assert (
        too_hard.metadata["frontier_semantic_signal"]
        == "conditional_on_successful_execution"
    )


def _syntax_ready_solver() -> SolverMilestones:
    return SolverMilestones(
        output_present=1.0,
        json_valid=1.0,
        schema_fraction=1.0,
        valid_operation_fraction=1.0,
        valid_prefix_fraction=1.0,
    )


def test_solver_syntax_reward_distinguishes_partial_outputs() -> None:
    partial = solver_curriculum_reward(
        SolverMilestones(output_present=1.0, json_valid=1.0, schema_fraction=0.5),
        stage="syntax",
    )
    complete = solver_curriculum_reward(_syntax_ready_solver(), stage="syntax")

    assert 0.0 < partial.total < complete.total == pytest.approx(1.0)
    assert partial.components["milestone_schema_fraction"] == 0.5
    assert partial.components["stage_syntax"] == 1.0


def test_solver_tool_reward_uses_call_success_before_answer_success() -> None:
    syntax = _syntax_ready_solver()
    invalid_calls = solver_curriculum_reward(
        replace(
            syntax,
            tool_call_attempted=1.0,
            valid_tool_call_fraction=0.25,
            successful_tool_call_fraction=0.0,
            budget_compliance=1.0,
        ),
        stage="tool",
    )
    useful_calls = solver_curriculum_reward(
        replace(
            syntax,
            tool_call_attempted=1.0,
            valid_tool_call_fraction=1.0,
            successful_tool_call_fraction=0.75,
            evidence_progress=0.5,
            execution_progress=0.5,
            budget_compliance=1.0,
        ),
        stage="tool",
    )

    assert useful_calls.total > invalid_calls.total
    assert useful_calls.components["tool_contribution"] > 0.0
    assert useful_calls.components["solve_contribution"] == 0.0


def test_solver_solve_reward_is_dense_in_f1_and_exact_match() -> None:
    ready = replace(
        _syntax_ready_solver(),
        tool_call_attempted=1.0,
        valid_tool_call_fraction=1.0,
        successful_tool_call_fraction=1.0,
        evidence_progress=1.0,
        execution_progress=1.0,
        budget_compliance=1.0,
        answer_present=1.0,
        answer_parse_valid=1.0,
    )
    wrong = solver_curriculum_reward(ready, stage="solve")
    partial = solver_curriculum_reward(
        replace(ready, answer_f1=0.5), stage="solve"
    )
    exact = solver_curriculum_reward(
        replace(ready, answer_f1=1.0, exact_match=1.0), stage="solve"
    )

    assert wrong.total < partial.total < exact.total == pytest.approx(1.0)
    assert partial.components["milestone_answer_f1"] == 0.5
    assert partial.components["solve_contribution"] > 0.0
    assert partial.components["curriculum_total"] == partial.total


def test_curriculum_inputs_reject_non_normalized_metrics() -> None:
    with pytest.raises(ValueError, match="answer_f1"):
        SolverMilestones(answer_f1=1.1)

    with pytest.raises(ValueError, match="parse_rate"):
        OpponentSignals(
            parse_rate=-0.1,
            execution_rate_given_parse=1.0,
            semantic_success_given_execution=0.5,
        )
