from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from graphtask_r1.dsl import canonical_signature
from graphtask_r1.graph import GraphBackend
from graphtask_r1.graphscript import (
    GraphScriptError,
    execute_graphscript,
    parse_graphscript,
    program_to_graphscript,
)
from graphtask_r1.schema import AnswerSet, RelationInfo, TaskTrainingRecord
from graphtask_r1.utils import stable_hash

DEFAULT_SLOT_PRIORITY = (
    "relation",
    "direction",
    "qualifier",
    "attribute",
    "comparator",
    "literal",
    "mode",
    "set_operator",
    "type_id",
    "query",
)

_COMPARATOR_COUNTERFACTUAL = {
    "eq": "ne",
    "ne": "eq",
    "lt": "ge",
    "le": "gt",
    "gt": "le",
    "ge": "lt",
}


@dataclass(frozen=True)
class SlotEdit:
    operation_index: int
    operation: str
    slot: str
    old_value: object
    new_value: object

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_index": self.operation_index,
            "operation": self.operation,
            "slot": self.slot,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


@dataclass(frozen=True)
class CounterfactualCertificate:
    task_id: str
    chosen_graphscript: str
    rejected_graphscript: str
    chosen_program_signature: str
    rejected_program_signature: str
    chosen_answers: tuple[str, ...]
    rejected_answers: tuple[str, ...]
    edit: SlotEdit

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "chosen_graphscript": self.chosen_graphscript,
            "rejected_graphscript": self.rejected_graphscript,
            "chosen_program_signature": self.chosen_program_signature,
            "rejected_program_signature": self.rejected_program_signature,
            "chosen_answers": list(self.chosen_answers),
            "rejected_answers": list(self.rejected_answers),
            "edit": self.edit.to_dict(),
            "single_slot": True,
            "both_executable": True,
            "denotation_changed": True,
        }


@dataclass(frozen=True)
class MutationOutcome:
    certificate: CounterfactualCertificate | None
    reason: str
    attempt_rejections: tuple[MutationAttemptRejection, ...] = ()


AttemptStage = Literal[
    "candidate_parse",
    "single_slot_validation",
    "candidate_execution",
    "denotation_validation",
]


@dataclass(frozen=True)
class MutationAttemptRejection:
    """One rejected candidate without retaining the raw proposed value."""

    task_id: str
    slot: str
    operation_index: int
    value_sha256: str
    stage: AttemptStage
    reason_code: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "slot": self.slot,
            "operation_index": self.operation_index,
            "value_sha256": self.value_sha256,
            "stage": self.stage,
            "reason_code": self.reason_code,
        }


def _answer_keys(answers: AnswerSet) -> tuple[str, ...]:
    return tuple(f"{answer.kind}:{answer.value}" for answer in answers.answers)


class CounterfactualMutator:
    """Create one-slot, dual-executed GraphScript counterfactuals.

    The mutator never edits the certified task.  It copies its flattened GraphScript,
    changes exactly one semantic slot, and accepts the negative only when both programs
    execute under the same budget and return different, non-empty denotations.
    """

    def __init__(
        self,
        backend: GraphBackend,
        relations: tuple[RelationInfo, ...],
        *,
        seed: int,
        max_edge_visits: int,
        max_returned_entities: int,
        observed_values: dict[str, tuple[object, ...]] | None = None,
    ) -> None:
        self.backend = backend
        self.relations = relations
        self.seed = seed
        self.max_edge_visits = max_edge_visits
        self.max_returned_entities = max_returned_entities
        self.allowed_relations = frozenset(relation.relation_id for relation in relations)
        self.relation_by_id = {relation.relation_id: relation for relation in relations}
        self.observed_values = observed_values or {}

    def mutate(
        self,
        task: TaskTrainingRecord,
        *,
        slot_priority: tuple[str, ...] = DEFAULT_SLOT_PRIORITY,
    ) -> MutationOutcome:
        if not task.verification.executable:
            return MutationOutcome(None, "CHOSEN_NOT_CERTIFIED")
        try:
            chosen = program_to_graphscript(task.program, version="0.3")
            chosen_execution = execute_graphscript(
                chosen,
                self.backend,
                allowed_relations=self.allowed_relations,
                max_edge_visits=self.max_edge_visits,
                max_returned_entities=self.max_returned_entities,
                trace_id=f"ere:{task.task_id}:chosen",
            )
        except GraphScriptError as exc:
            return MutationOutcome(None, f"CHOSEN_{exc.reason_code}")
        except (KeyError, TypeError, ValueError):
            return MutationOutcome(None, "CHOSEN_CONVERSION_FAILED")
        if not chosen_execution.answers.answers:
            return MutationOutcome(None, "CHOSEN_EMPTY")
        if chosen_execution.answers != task.gold_answers:
            return MutationOutcome(None, "CHOSEN_GOLD_MISMATCH")

        attempt_rejections: list[MutationAttemptRejection] = []

        def reject_attempt(
            *,
            slot: str,
            operation_index: int,
            new_value: object,
            stage: AttemptStage,
            reason_code: str,
        ) -> None:
            attempt_rejections.append(
                MutationAttemptRejection(
                    task_id=task.task_id,
                    slot=slot,
                    operation_index=operation_index,
                    value_sha256=stable_hash(new_value),
                    stage=stage,
                    reason_code=reason_code,
                )
            )

        raw_chosen = chosen.model_dump(mode="json", by_alias=True)
        raw_ops = cast(list[dict[str, Any]], raw_chosen["ops"])
        for slot in slot_priority:
            for operation_index, raw_operation in enumerate(raw_ops):
                for new_value in self._slot_values(
                    task.task_id,
                    operation_index,
                    raw_operation,
                    slot,
                ):
                    candidate_operation = dict(raw_operation)
                    field = {
                        "set_operator": "op",
                        "literal": "value",
                    }.get(slot, slot)
                    old_value = candidate_operation.get(field)
                    if old_value == new_value:
                        continue
                    candidate_operation[field] = new_value
                    candidate_ops = [dict(value) for value in raw_ops]
                    candidate_ops[operation_index] = candidate_operation
                    candidate_payload = {"version": "0.3", "ops": candidate_ops}
                    try:
                        rejected = parse_graphscript(
                            candidate_payload,
                            max_follow_limit=100,
                        )
                    except GraphScriptError as exc:
                        reject_attempt(
                            slot=slot,
                            operation_index=operation_index,
                            new_value=new_value,
                            stage="candidate_parse",
                            reason_code=exc.reason_code,
                        )
                        continue
                    rejected_payload = rejected.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                    rejected_ops = cast(
                        list[dict[str, Any]],
                        rejected_payload["ops"],
                    )
                    if _operation_differences(raw_ops, rejected_ops) != ((operation_index, field),):
                        reject_attempt(
                            slot=slot,
                            operation_index=operation_index,
                            new_value=new_value,
                            stage="single_slot_validation",
                            reason_code="NOT_SINGLE_TOP_LEVEL_SLOT",
                        )
                        continue
                    try:
                        rejected_execution = execute_graphscript(
                            rejected,
                            self.backend,
                            allowed_relations=self.allowed_relations,
                            max_edge_visits=self.max_edge_visits,
                            max_returned_entities=self.max_returned_entities,
                            trace_id=(f"ere:{task.task_id}:rejected:{operation_index}:{slot}"),
                        )
                    except GraphScriptError as exc:
                        reject_attempt(
                            slot=slot,
                            operation_index=operation_index,
                            new_value=new_value,
                            stage="candidate_execution",
                            reason_code=exc.reason_code,
                        )
                        continue
                    if not rejected_execution.answers.answers:
                        reject_attempt(
                            slot=slot,
                            operation_index=operation_index,
                            new_value=new_value,
                            stage="denotation_validation",
                            reason_code="EMPTY_RESULT",
                        )
                        continue
                    if rejected_execution.answers == chosen_execution.answers:
                        reject_attempt(
                            slot=slot,
                            operation_index=operation_index,
                            new_value=new_value,
                            stage="denotation_validation",
                            reason_code="UNCHANGED_DENOTATION",
                        )
                        continue
                    edit = SlotEdit(
                        operation_index=operation_index,
                        operation=str(raw_operation.get("op", "")),
                        slot=slot,
                        old_value=old_value,
                        new_value=new_value,
                    )
                    certificate = CounterfactualCertificate(
                        task_id=task.task_id,
                        chosen_graphscript=chosen.model_dump_json(by_alias=True),
                        rejected_graphscript=rejected.model_dump_json(by_alias=True),
                        chosen_program_signature=canonical_signature(chosen_execution.program),
                        rejected_program_signature=canonical_signature(rejected_execution.program),
                        chosen_answers=_answer_keys(chosen_execution.answers),
                        rejected_answers=_answer_keys(rejected_execution.answers),
                        edit=edit,
                    )
                    return MutationOutcome(
                        certificate,
                        "ACCEPTED",
                        tuple(attempt_rejections),
                    )
        return MutationOutcome(
            None,
            "NO_DIVERGENT_SINGLE_SLOT_MUTANT",
            tuple(attempt_rejections),
        )

    def _slot_values(
        self,
        task_id: str,
        operation_index: int,
        operation: dict[str, Any],
        slot: str,
    ) -> tuple[object, ...]:
        field = {"set_operator": "op", "literal": "value"}.get(slot, slot)
        if slot == "set_operator":
            if operation.get("op") not in {"intersect", "union"}:
                return ()
        elif field not in operation:
            return ()
        values: list[object] = list(self.observed_values.get(slot, ()))
        current = operation.get(field)
        if slot == "direction" and current in {"out", "in"}:
            values.append("in" if current == "out" else "out")
        elif slot == "comparator" and current in _COMPARATOR_COUNTERFACTUAL:
            values.append(_COMPARATOR_COUNTERFACTUAL[str(current)])
        elif slot == "mode" and current in {"min", "max"}:
            values.append("max" if current == "min" else "min")
        elif slot == "set_operator":
            if operation.get("op") == "intersect":
                values.append("union")
            elif operation.get("op") == "union":
                values.append("intersect")
        elif slot in {"relation", "attribute", "qualifier"} and isinstance(current, str):
            values.extend(
                self._compatible_relations(
                    current,
                    task_id=task_id,
                    operation_index=operation_index,
                    slot=slot,
                )
            )
        unique: dict[str, object] = {}
        for value in values:
            if value != current:
                unique.setdefault(stable_hash(value), value)
        return tuple(unique.values())

    def _compatible_relations(
        self,
        relation_id: str,
        *,
        task_id: str,
        operation_index: int,
        slot: str,
    ) -> tuple[str, ...]:
        reference = self.relation_by_id.get(relation_id)
        if reference is None:
            candidates = [relation.relation_id for relation in self.relations]
        else:
            candidates = [
                relation.relation_id
                for relation in self.relations
                if relation.relation_id != relation_id
                and relation.domain_types == reference.domain_types
                and relation.range_types == reference.range_types
            ]
        ranked = sorted(
            candidates,
            key=lambda candidate: stable_hash(
                [self.seed, task_id, operation_index, slot, candidate]
            ),
        )
        return tuple(ranked[:24])


def _operation_differences(
    chosen: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> tuple[tuple[int, str], ...]:
    differences: list[tuple[int, str]] = []
    for operation_index, (chosen_operation, rejected_operation) in enumerate(
        zip(chosen, rejected, strict=True)
    ):
        for field in sorted(chosen_operation.keys() | rejected_operation.keys()):
            if chosen_operation.get(field) != rejected_operation.get(field):
                differences.append((operation_index, field))
    return tuple(differences)
