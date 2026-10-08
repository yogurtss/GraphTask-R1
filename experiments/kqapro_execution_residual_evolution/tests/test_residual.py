from __future__ import annotations

import json

import pytest
from kqapro_ere import (
    Denotation,
    ExecutionBudget,
    OperationFields,
    PrefixDenotation,
    ResidualAnalysis,
    align_prefix_denotations,
    analyze_execution_residual,
)

from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Entity,
    FilterType,
    Hop,
    program_to_dict,
)


def _reference_program() -> str:
    program = Hop(
        input=Hop(
            input=Entity(entity_id="alice"),
            relation="works_at",
            direction="out",
        ),
        relation="located_in",
        direction="out",
    )
    return json.dumps(program_to_dict(program), sort_keys=True)


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
        },
        sort_keys=True,
    )


def _analyze(script: str) -> ResidualAnalysis:
    return analyze_execution_residual(
        script,
        _reference_program(),
        backend=toy_graph(),
        budget=ExecutionBudget(),
        allowed_relations=frozenset({"works_at", "located_in"}),
        seed_entity=None,
        seed=17,
        trace_id="residual-test",
    )


def _prefix(step: int, values: tuple[str, ...]) -> PrefixDenotation:
    operation: dict[str, object] = {
        "op": "resolve_entity",
        "query": f"q{step}",
        "match": "id",
        "limit": 1,
        "out": f"h{step}",
    }
    return PrefixDenotation(
        step_index=step,
        output_handle=f"h{step}",
        operation=operation,
        fields=OperationFields.from_operation(operation),
        denotation=Denotation.materialized(AnswerSet.entities(values)),
    )


def test_exact_execution_prefix_alignment_is_replayable() -> None:
    analysis = _analyze(_script())

    assert analysis.alignment.score == 1.0
    assert analysis.alignment.denominator == 3
    assert analysis.earliest_divergence is None
    assert analysis.predicted_terminal.answers.values() == ("paris",)
    assert analysis.gold_terminal.answers.values() == ("paris",)
    assert analysis.seed == 17
    assert analysis.budget == ExecutionBudget()
    json.dumps(analysis.model_dump(mode="json"), sort_keys=True)


def test_earliest_divergence_preserves_complete_gold_and_predicted_operations() -> None:
    analysis = _analyze(_script(final_relation="works_at", final_direction="in"))

    assert analysis.alignment.score == pytest.approx(2.0 / 3.0)
    residual = analysis.earliest_divergence
    assert residual is not None
    assert residual.reason == "denotation_mismatch"
    assert residual.gold_step_index == 2
    assert residual.predicted_step_index == 2
    assert residual.gold_op == "follow"
    assert residual.predicted_op == "follow"
    assert residual.gold_fields.relation == "located_in"
    assert residual.gold_fields.direction == "out"
    assert residual.predicted_fields is not None
    assert residual.predicted_fields.relation == "works_at"
    assert residual.predicted_fields.direction == "in"
    assert residual.gold_operation == {
        "op": "follow",
        "in": "h1",
        "relation": "located_in",
        "direction": "out",
        "limit": 100,
        "out": "h2",
    }
    assert residual.predicted_operation == {
        "op": "follow",
        "in": "h1",
        "relation": "works_at",
        "direction": "in",
        "limit": 100,
        "out": "h2",
    }


def test_missing_predicted_prefix_keeps_the_next_gold_operation_actionable() -> None:
    short_script = json.dumps(
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
                {"op": "emit", "in": "h0"},
            ],
        }
    )

    analysis = _analyze(short_script)

    residual = analysis.earliest_divergence
    assert residual is not None
    assert residual.reason == "missing_prediction"
    assert residual.gold_step_index == 1
    assert residual.gold_fields.relation == "works_at"
    assert residual.predicted_step_index is None
    assert residual.predicted_operation is None


def test_operation_fields_expose_all_mutation_keys_without_losing_payload() -> None:
    operation: dict[str, object] = {
        "op": "synthetic_test_operation",
        "relation": "member_of",
        "direction": "in",
        "attribute": "population",
        "qualifier": "point_in_time",
        "comparator": "ge",
        "out": "h7",
    }

    fields = OperationFields.from_operation(operation)

    assert fields.model_dump() == {
        "op": "synthetic_test_operation",
        "relation": "member_of",
        "direction": "in",
        "attribute": "population",
        "qualifier": "point_in_time",
        "comparator": "ge",
    }


def test_monotone_dp_uses_soft_denotation_f1_and_longer_sequence_denominator() -> None:
    predicted = (
        _prefix(0, ("a", "b")),
        _prefix(1, ("c",)),
        _prefix(2, ("unused",)),
    )
    gold = (_prefix(0, ("a",)), _prefix(1, ("c",)))

    alignment = align_prefix_denotations(predicted, gold)

    assert alignment.raw_similarity == pytest.approx(5.0 / 3.0)
    assert alignment.denominator == 3
    assert alignment.score == pytest.approx(5.0 / 9.0)
    assert [(pair.predicted_position, pair.gold_position) for pair in alignment.pairs] == [
        (0, 0),
        (1, 1),
    ]
    assert alignment.unmatched_predicted_step_indices == (0, 2)
    assert alignment.unmatched_gold_step_indices == (0,)


def test_all_entities_prefix_stays_deferred_but_downstream_prefix_executes() -> None:
    reference = json.dumps(
        program_to_dict(
            FilterType(
                input=AllEntities(max_results=100),
                type_id="person",
            )
        )
    )
    predicted = json.dumps(
        {
            "version": "0.3",
            "ops": [
                {"op": "all_entities", "max_results": 100, "out": "h0"},
                {"op": "filter_type", "in": "h0", "type_id": "person", "out": "h1"},
                {"op": "emit", "in": "h1"},
            ],
        }
    )

    analysis = analyze_execution_residual(
        predicted,
        reference,
        backend=toy_graph(),
        budget=ExecutionBudget(),
        allowed_relations=frozenset(),
        seed_entity=None,
        seed=23,
        trace_id="all-entities-test",
    )

    root = analysis.predicted_prefixes[0]
    downstream = analysis.predicted_prefixes[1]
    assert root.denotation.state == "deferred"
    assert root.denotation.answers is None
    assert root.denotation.deferred_max_results == 100
    assert downstream.denotation.answers is not None
    assert downstream.denotation.answers.values() == ("alice", "bob", "cara")
    assert analysis.alignment.score == 1.0
