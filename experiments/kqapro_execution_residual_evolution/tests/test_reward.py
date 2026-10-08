from __future__ import annotations

import json

import pytest
from kqapro_ere import (
    ERERewardConfig,
    ERERewardResult,
    ExecutionBudget,
    compute_ere_reward,
)

from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import Entity, Hop, program_to_dict


def _reference_program() -> str:
    return json.dumps(
        program_to_dict(
            Hop(
                input=Hop(
                    input=Entity(entity_id="alice"),
                    relation="works_at",
                    direction="out",
                ),
                relation="located_in",
                direction="out",
            )
        )
    )


def _script(*, final_relation: str = "located_in", final_direction: str = "out") -> str:
    return json.dumps(
        {
            "version": "0.3",
            "ops": [
                {
                    "op": "resolve_entity",
                    "query": "alice",
                    "match": "id",
                    "limit": 1,
                    "out": "h0",
                },
                {
                    "op": "follow",
                    "in": "h0",
                    "relation": "works_at",
                    "direction": "out",
                    "limit": 100,
                    "out": "h1",
                },
                {
                    "op": "follow",
                    "in": "h1",
                    "relation": final_relation,
                    "direction": final_direction,
                    "limit": 100,
                    "out": "h2",
                },
                {"op": "emit", "in": "h2"},
            ],
        }
    )


def _score(solution: str) -> ERERewardResult:
    return compute_ere_reward(
        solution,
        _reference_program(),
        backend=toy_graph(),
        config=ERERewardConfig(
            budget=ExecutionBudget(max_edge_visits=10),
            terminal_bonus_scale=0.25,
        ),
        allowed_relations=frozenset({"works_at", "located_in"}),
        seed_entity=None,
        seed=31,
        trace_id="reward-test",
    )


def test_reward_is_one_alignment_potential_plus_terminal_bonus() -> None:
    result = _score(_script())

    assert result.accepted
    assert result.rejection is None
    assert result.components.alignment_potential == 1.0
    assert result.components.terminal_answer_f1 == 1.0
    assert result.components.terminal_exact_match == 1.0
    assert result.components.terminal_bonus == 0.25
    assert result.total == 1.25
    assert result.analysis is not None
    assert result.analysis.predicted_terminal.usage.edge_visits == 2
    assert result.analysis.gold_terminal.usage.edge_visits == 2
    json.dumps(result.model_dump(mode="json"), sort_keys=True)


def test_reward_keeps_prefix_localization_when_terminal_answer_is_wrong() -> None:
    result = _score(_script(final_relation="works_at", final_direction="in"))

    assert result.accepted
    assert result.components.alignment_potential == pytest.approx(2.0 / 3.0)
    assert result.components.terminal_answer_f1 == 0.0
    assert result.components.terminal_exact_match == 0.0
    assert result.components.terminal_bonus == 0.0
    assert result.total == pytest.approx(2.0 / 3.0)
    assert result.analysis is not None
    assert result.analysis.earliest_divergence is not None
    assert result.analysis.earliest_divergence.gold_fields.relation == "located_in"


def test_empty_denotation_remains_an_actionable_execution_residual() -> None:
    result = _score(_script(final_relation="located_in", final_direction="in"))

    assert result.accepted
    assert result.rejection is None
    assert result.analysis is not None
    assert result.analysis.predicted_terminal.answers.answers == ()
    assert result.analysis.predicted_terminal.usage is None
    assert result.analysis.earliest_divergence is not None
    assert result.components.terminal_exact_match == 0.0


@pytest.mark.parametrize(
    ("solution", "reason"),
    [
        ("not-json", "NON_JSON"),
        (_script() + " trailing", "EXTRA_TEXT"),
    ],
)
def test_reward_strict_parser_returns_structured_rejection(
    solution: str,
    reason: str,
) -> None:
    result = _score(solution)

    assert not result.accepted
    assert result.total == 0.0
    assert result.analysis is None
    assert result.rejection is not None
    assert result.rejection.reason_code == reason
    assert result.rejection.stage == "predicted_parse"
    assert result.components.alignment_potential == 0.0


def test_reference_json_is_also_strict() -> None:
    result = compute_ere_reward(
        _script(),
        _reference_program() + " trailing",
        backend=toy_graph(),
        config=ERERewardConfig(),
        allowed_relations=frozenset({"works_at", "located_in"}),
        seed_entity=None,
        seed=37,
        trace_id="strict-reference-test",
    )

    assert not result.accepted
    assert result.rejection is not None
    assert result.rejection.reason_code == "EXTRA_TEXT"
    assert result.rejection.stage == "reference_parse"
