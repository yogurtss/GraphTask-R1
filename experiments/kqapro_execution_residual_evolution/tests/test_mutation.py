from __future__ import annotations

import json

from kqapro_ere.mutation import CounterfactualMutator

from graphtask_r1.graph import toy_graph
from graphtask_r1.graphscript import execute_graphscript, parse_graphscript
from graphtask_r1.schema import (
    AnswerSet,
    Entity,
    EntityInfo,
    Hop,
    TaskTrainingRecord,
    VerificationSummary,
)
from graphtask_r1.utils import stable_hash


def _friend_task(
    *,
    executable: bool = True,
    gold_answers: AnswerSet | None = None,
) -> TaskTrainingRecord:
    return TaskTrainingRecord(
        task_id="toy-friend",
        question="Who is Bob's friend?",
        topic_entities=(EntityInfo(entity_id="bob", label="Bob"),),
        program=Hop(input=Entity(entity_id="bob"), relation="friend"),
        gold_answers=gold_answers or AnswerSet.entities(["cara"]),
        verification=VerificationSummary(executable=executable),
    )


def _mutator(seed: int) -> CounterfactualMutator:
    backend = toy_graph()
    return CounterfactualMutator(
        backend,
        tuple(backend.all_relation_infos()),
        seed=seed,
        max_edge_visits=100,
        max_returned_entities=100,
    )


def test_mutation_requires_certified_executed_gold() -> None:
    not_certified = _mutator(7).mutate(_friend_task(executable=False))
    wrong_gold = _mutator(7).mutate(_friend_task(gold_answers=AnswerSet.entities(["alice"])))

    assert not_certified.certificate is None
    assert not_certified.reason == "CHOSEN_NOT_CERTIFIED"
    assert wrong_gold.certificate is None
    assert wrong_gold.reason == "CHOSEN_GOLD_MISMATCH"


def test_direction_mutation_is_one_slot_dual_executed_and_answer_divergent() -> None:
    backend = toy_graph()
    relations = tuple(backend.all_relation_infos())
    outcome = CounterfactualMutator(
        backend,
        relations,
        seed=23,
        max_edge_visits=100,
        max_returned_entities=100,
    ).mutate(_friend_task(), slot_priority=("direction",))

    assert outcome.reason == "ACCEPTED"
    assert outcome.certificate is not None
    certificate = outcome.certificate
    chosen_payload = json.loads(certificate.chosen_graphscript)
    rejected_payload = json.loads(certificate.rejected_graphscript)
    differences: list[tuple[int, str]] = []
    for index, (chosen_op, rejected_op) in enumerate(
        zip(chosen_payload["ops"], rejected_payload["ops"], strict=True)
    ):
        for field in sorted(chosen_op.keys() | rejected_op.keys()):
            if chosen_op.get(field) != rejected_op.get(field):
                differences.append((index, field))
    assert differences == [(certificate.edit.operation_index, "direction")]
    assert certificate.edit.old_value == "out"
    assert certificate.edit.new_value == "in"

    allowed_relations = frozenset(relation.relation_id for relation in relations)
    chosen = execute_graphscript(
        parse_graphscript(certificate.chosen_graphscript),
        backend,
        allowed_relations=allowed_relations,
        max_edge_visits=100,
        max_returned_entities=100,
        trace_id="test:chosen",
    )
    rejected = execute_graphscript(
        parse_graphscript(certificate.rejected_graphscript),
        backend,
        allowed_relations=allowed_relations,
        max_edge_visits=100,
        max_returned_entities=100,
        trace_id="test:rejected",
    )
    assert chosen.answers == _friend_task().gold_answers
    assert chosen.answers.answers
    assert rejected.answers.answers
    assert chosen.answers != rejected.answers
    assert certificate.to_dict()["both_executable"] is True
    assert certificate.to_dict()["denotation_changed"] is True


def test_relation_candidates_are_seeded_and_replayable() -> None:
    first = _mutator(0).mutate(_friend_task(), slot_priority=("relation",))
    replay = _mutator(0).mutate(_friend_task(), slot_priority=("relation",))
    other_seed = _mutator(3).mutate(_friend_task(), slot_priority=("relation",))

    assert first.certificate is not None
    assert replay.certificate is not None
    assert other_seed.certificate is not None
    assert first.certificate.rejected_graphscript == replay.certificate.rejected_graphscript
    assert first.certificate.rejected_graphscript != other_seed.certificate.rejected_graphscript


def test_candidate_rejections_keep_structured_hash_only_ledger() -> None:
    backend = toy_graph()
    outcome = CounterfactualMutator(
        backend,
        tuple(backend.all_relation_infos()),
        seed=23,
        max_edge_visits=100,
        max_returned_entities=100,
        observed_values={"direction": ("sideways",)},
    ).mutate(_friend_task(), slot_priority=("direction",))

    assert outcome.certificate is not None
    assert len(outcome.attempt_rejections) == 1
    rejection = outcome.attempt_rejections[0]
    assert rejection.task_id == "toy-friend"
    assert rejection.slot == "direction"
    assert rejection.operation_index == 1
    assert rejection.value_sha256 == stable_hash("sideways")
    assert rejection.stage == "candidate_parse"
    assert rejection.reason_code == "INVALID_DIRECTION"
    assert set(rejection.to_dict()) == {
        "task_id",
        "slot",
        "operation_index",
        "value_sha256",
        "stage",
        "reason_code",
    }
    assert "sideways" not in json.dumps(rejection.to_dict())
