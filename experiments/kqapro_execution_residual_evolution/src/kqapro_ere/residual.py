from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from graphtask_r1.evaluation.answer_metrics import answer_metrics
from graphtask_r1.graph.base import GraphBackend
from graphtask_r1.graphscript import (
    AllEntitiesOp,
    BudgetUsage,
    EmitOp,
    GraphScript,
    GraphScriptError,
    execute_graphscript,
    parse_graphscript,
    program_to_graphscript,
)
from graphtask_r1.schema import AnswerSet, parse_program

GraphScriptVersion = Literal["0.1", "0.2", "0.3"]
DenotationState = Literal["materialized", "deferred"]


class ResidualError(ValueError):
    """A structured failure raised while constructing execution residuals."""

    def __init__(self, reason_code: str, detail: str, *, stage: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail
        self.stage = stage


class ExecutionBudget(BaseModel):
    """The one budget contract used for both candidate and certified executions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graphscript_version: GraphScriptVersion = "0.3"
    max_follow_limit: int = Field(default=100, gt=0)
    max_edge_visits: int = Field(default=200, gt=0)
    max_returned_entities: int = Field(default=1_000, gt=0)


class OperationFields(BaseModel):
    """Mutation-relevant fields beside the lossless operation payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: str
    relation: str | None = None
    direction: str | None = None
    attribute: str | None = None
    qualifier: str | None = None
    comparator: str | None = None

    @classmethod
    def from_operation(cls, operation: dict[str, object]) -> OperationFields:
        def optional_text(name: str) -> str | None:
            value = operation.get(name)
            return str(value) if value is not None else None

        return cls(
            op=str(operation["op"]),
            relation=optional_text("relation"),
            direction=optional_text("direction"),
            attribute=optional_text("attribute"),
            qualifier=optional_text("qualifier"),
            comparator=optional_text("comparator"),
        )


class Denotation(BaseModel):
    """A materialized AnswerSet or the bounded all-entities deferred sentinel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: DenotationState
    answers: AnswerSet | None = None
    deferred_op: Literal["all_entities"] | None = None
    deferred_max_results: int | None = None

    @classmethod
    def materialized(cls, answers: AnswerSet) -> Denotation:
        return cls(state="materialized", answers=answers)

    @classmethod
    def all_entities(cls, max_results: int) -> Denotation:
        return cls(
            state="deferred",
            deferred_op="all_entities",
            deferred_max_results=max_results,
        )


class PrefixDenotation(BaseModel):
    """The complete operation and execution denotation at one output prefix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=0)
    output_handle: str
    operation: dict[str, object]
    fields: OperationFields
    denotation: Denotation
    usage: BudgetUsage | None = None


class AlignmentPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predicted_position: int = Field(ge=0)
    gold_position: int = Field(ge=0)
    predicted_step_index: int = Field(ge=0)
    gold_step_index: int = Field(ge=0)
    similarity: float = Field(ge=0.0, le=1.0)
    exact_match: bool


class DenotationAlignment(BaseModel):
    """A monotone, one-to-one alignment normalized by the longer sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    raw_similarity: float = Field(ge=0.0)
    denominator: int = Field(ge=0)
    exact_matches: int = Field(ge=0)
    pairs: tuple[AlignmentPair, ...]
    unmatched_predicted_step_indices: tuple[int, ...]
    unmatched_gold_step_indices: tuple[int, ...]


class ExecutionResidual(BaseModel):
    """The earliest gold prefix without an exact denotation match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: Literal["missing_prediction", "denotation_mismatch"]
    gold_step_index: int = Field(ge=0)
    predicted_step_index: int | None
    gold_op: str
    predicted_op: str | None
    gold_operation: dict[str, object]
    predicted_operation: dict[str, object] | None
    gold_fields: OperationFields
    predicted_fields: OperationFields | None
    gold_denotation: Denotation
    predicted_denotation: Denotation | None
    denotation_similarity: float = Field(ge=0.0, le=1.0)


class TerminalExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answers: AnswerSet
    usage: BudgetUsage | None


class ResidualAnalysis(BaseModel):
    """Replayable evidence for residual-guided single-slot mutation or ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    trace_id: str
    budget: ExecutionBudget
    predicted_graphscript: dict[str, object]
    gold_graphscript: dict[str, object]
    predicted_prefixes: tuple[PrefixDenotation, ...]
    gold_prefixes: tuple[PrefixDenotation, ...]
    alignment: DenotationAlignment
    earliest_divergence: ExecutionResidual | None
    predicted_terminal: TerminalExecution
    gold_terminal: TerminalExecution
    terminal_answer_f1: float = Field(ge=0.0, le=1.0)
    terminal_exact_match: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class _DPCell:
    score: float
    exact_matches: int
    pair_count: int
    choice: Literal["start", "skip_predicted", "skip_gold", "match"]


def _strict_json(text: str, *, stage: str) -> object:
    stripped = text.strip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ResidualError("NON_JSON", str(exc), stage=stage) from exc
    if stripped[end:].strip():
        raise ResidualError(
            "EXTRA_TEXT",
            "input must contain exactly one JSON value",
            stage=stage,
        )
    return value


def _script_payload(script: GraphScript) -> dict[str, object]:
    return cast(
        dict[str, object],
        script.model_dump(mode="json", by_alias=True),
    )


def _operation_payload(operation: BaseModel) -> dict[str, object]:
    return cast(
        dict[str, object],
        operation.model_dump(mode="json", by_alias=True),
    )


def parse_reference_graphscript(
    reference_program_json: str,
    *,
    backend: GraphBackend,
    budget: ExecutionBudget,
) -> GraphScript:
    """Compile the hidden certified Program into the shared GraphScript executor profile."""

    raw_program = _strict_json(reference_program_json, stage="reference_parse")
    try:
        program = parse_program(raw_program)
    except ValidationError as exc:
        raise ResidualError(
            "INVALID_REFERENCE_PROGRAM",
            str(exc),
            stage="reference_parse",
        ) from exc
    try:
        return program_to_graphscript(
            program,
            follow_limit=budget.max_follow_limit,
            version=budget.graphscript_version,
            entity_reference=lambda entity_id: (entity_id, "id"),
        )
    except GraphScriptError as exc:
        raise ResidualError(
            exc.reason_code,
            exc.detail,
            stage="reference_compile",
        ) from exc


def _execute_terminal(
    script: GraphScript,
    *,
    backend: GraphBackend,
    budget: ExecutionBudget,
    seed_entity: str | None,
    allowed_relations: frozenset[str],
    trace_id: str,
    stage: str,
) -> TerminalExecution:
    try:
        execution = execute_graphscript(
            script,
            backend,
            seed_entity=seed_entity,
            allowed_relations=allowed_relations,
            max_edge_visits=budget.max_edge_visits,
            max_returned_entities=budget.max_returned_entities,
            trace_id=trace_id,
        )
    except GraphScriptError as exc:
        if exc.reason_code == "EMPTY_RESULT":
            return TerminalExecution(answers=AnswerSet(), usage=None)
        raise ResidualError(exc.reason_code, exc.detail, stage=stage) from exc
    return TerminalExecution(answers=execution.answers, usage=execution.usage)


def execute_output_prefixes(
    script: GraphScript,
    *,
    backend: GraphBackend,
    budget: ExecutionBudget,
    seed_entity: str | None,
    allowed_relations: frozenset[str],
    trace_id: str,
) -> tuple[PrefixDenotation, ...]:
    """Execute every handle-producing prefix under one immutable budget contract."""

    prefixes: list[PrefixDenotation] = []
    for step_index, operation in enumerate(script.ops):
        output_handle = getattr(operation, "out", None)
        if not isinstance(output_handle, str):
            continue
        operation_payload = _operation_payload(operation)
        if isinstance(operation, AllEntitiesOp):
            prefixes.append(
                PrefixDenotation(
                    step_index=step_index,
                    output_handle=output_handle,
                    operation=operation_payload,
                    fields=OperationFields.from_operation(operation_payload),
                    denotation=Denotation.all_entities(operation.max_results),
                )
            )
            continue

        prefix_script = GraphScript.model_validate(
            {
                "version": script.version,
                "ops": [
                    *(
                        prefix_operation.model_dump(mode="json", by_alias=True)
                        for prefix_operation in script.ops[: step_index + 1]
                    ),
                    EmitOp.model_validate({"in": output_handle}).model_dump(
                        mode="json", by_alias=True
                    ),
                ],
            }
        )
        try:
            execution = execute_graphscript(
                prefix_script,
                backend,
                seed_entity=seed_entity,
                allowed_relations=allowed_relations,
                max_edge_visits=budget.max_edge_visits,
                max_returned_entities=budget.max_returned_entities,
                trace_id=f"{trace_id}:prefix:{step_index}",
            )
            denotation = Denotation.materialized(execution.answers)
            usage: BudgetUsage | None = execution.usage
        except GraphScriptError as exc:
            if exc.reason_code != "EMPTY_RESULT":
                raise ResidualError(
                    exc.reason_code,
                    exc.detail,
                    stage="prefix_execution",
                ) from exc
            denotation = Denotation.materialized(AnswerSet())
            usage = None
        prefixes.append(
            PrefixDenotation(
                step_index=step_index,
                output_handle=output_handle,
                operation=operation_payload,
                fields=OperationFields.from_operation(operation_payload),
                denotation=denotation,
                usage=usage,
            )
        )
    return tuple(prefixes)


def _denotation_similarity(left: Denotation, right: Denotation) -> tuple[float, bool]:
    if left.state == "deferred" or right.state == "deferred":
        exact = (
            left.state == right.state
            and left.deferred_op == right.deferred_op
            and left.deferred_max_results == right.deferred_max_results
        )
        return float(exact), exact
    if left.answers is None or right.answers is None:
        return 0.0, False
    metrics = answer_metrics(left.answers, right.answers)
    return float(metrics["f1"]), bool(metrics["exact_match"])


def _cell_rank(cell: _DPCell, *, choice_priority: int) -> tuple[float, int, int, int]:
    return (cell.score, cell.exact_matches, cell.pair_count, choice_priority)


def align_prefix_denotations(
    predicted: tuple[PrefixDenotation, ...],
    gold: tuple[PrefixDenotation, ...],
) -> DenotationAlignment:
    """Find the maximum-similarity monotone alignment with zero-similarity pairs omitted."""

    predicted_count = len(predicted)
    gold_count = len(gold)
    grid: list[list[_DPCell]] = [
        [
            _DPCell(score=0.0, exact_matches=0, pair_count=0, choice="start")
            for _ in range(gold_count + 1)
        ]
        for _ in range(predicted_count + 1)
    ]
    for predicted_index in range(1, predicted_count + 1):
        grid[predicted_index][0] = _DPCell(
            score=0.0,
            exact_matches=0,
            pair_count=0,
            choice="skip_predicted",
        )
    for gold_index in range(1, gold_count + 1):
        grid[0][gold_index] = _DPCell(
            score=0.0,
            exact_matches=0,
            pair_count=0,
            choice="skip_gold",
        )

    for predicted_index in range(1, predicted_count + 1):
        for gold_index in range(1, gold_count + 1):
            skip_predicted_source = grid[predicted_index - 1][gold_index]
            skip_gold_source = grid[predicted_index][gold_index - 1]
            candidates = [
                (
                    _DPCell(
                        score=skip_predicted_source.score,
                        exact_matches=skip_predicted_source.exact_matches,
                        pair_count=skip_predicted_source.pair_count,
                        choice="skip_predicted",
                    ),
                    1,
                ),
                (
                    _DPCell(
                        score=skip_gold_source.score,
                        exact_matches=skip_gold_source.exact_matches,
                        pair_count=skip_gold_source.pair_count,
                        choice="skip_gold",
                    ),
                    0,
                ),
            ]
            similarity, exact = _denotation_similarity(
                predicted[predicted_index - 1].denotation,
                gold[gold_index - 1].denotation,
            )
            if similarity > 0.0:
                match_source = grid[predicted_index - 1][gold_index - 1]
                candidates.append(
                    (
                        _DPCell(
                            score=match_source.score + similarity,
                            exact_matches=match_source.exact_matches + int(exact),
                            pair_count=match_source.pair_count + 1,
                            choice="match",
                        ),
                        2,
                    )
                )
            grid[predicted_index][gold_index] = max(
                candidates,
                key=lambda candidate: _cell_rank(
                    candidate[0],
                    choice_priority=candidate[1],
                ),
            )[0]

    reversed_pairs: list[AlignmentPair] = []
    predicted_index = predicted_count
    gold_index = gold_count
    while predicted_index > 0 or gold_index > 0:
        choice = grid[predicted_index][gold_index].choice
        if choice == "match":
            predicted_prefix = predicted[predicted_index - 1]
            gold_prefix = gold[gold_index - 1]
            similarity, exact = _denotation_similarity(
                predicted_prefix.denotation,
                gold_prefix.denotation,
            )
            reversed_pairs.append(
                AlignmentPair(
                    predicted_position=predicted_index - 1,
                    gold_position=gold_index - 1,
                    predicted_step_index=predicted_prefix.step_index,
                    gold_step_index=gold_prefix.step_index,
                    similarity=similarity,
                    exact_match=exact,
                )
            )
            predicted_index -= 1
            gold_index -= 1
        elif choice == "skip_predicted":
            predicted_index -= 1
        elif choice == "skip_gold":
            gold_index -= 1
        else:
            break

    pairs = tuple(reversed(reversed_pairs))
    exact_predicted_positions = {pair.predicted_position for pair in pairs if pair.exact_match}
    exact_gold_positions = {pair.gold_position for pair in pairs if pair.exact_match}
    denominator = max(predicted_count, gold_count)
    raw_similarity = grid[predicted_count][gold_count].score
    score = raw_similarity / denominator if denominator else 1.0
    return DenotationAlignment(
        score=score,
        raw_similarity=raw_similarity,
        denominator=denominator,
        exact_matches=grid[predicted_count][gold_count].exact_matches,
        pairs=pairs,
        unmatched_predicted_step_indices=tuple(
            prefix.step_index
            for position, prefix in enumerate(predicted)
            if position not in exact_predicted_positions
        ),
        unmatched_gold_step_indices=tuple(
            prefix.step_index
            for position, prefix in enumerate(gold)
            if position not in exact_gold_positions
        ),
    )


def _predicted_candidate_position(
    gold_position: int,
    alignment: DenotationAlignment,
    predicted_count: int,
) -> int | None:
    paired = next(
        (
            pair.predicted_position
            for pair in alignment.pairs
            if pair.gold_position == gold_position
        ),
        None,
    )
    if paired is not None:
        return paired
    before = [
        pair.predicted_position for pair in alignment.pairs if pair.gold_position < gold_position
    ]
    after = [
        pair.predicted_position for pair in alignment.pairs if pair.gold_position > gold_position
    ]
    lower = max(before, default=-1) + 1
    upper = min(after, default=predicted_count)
    return lower if lower < upper else None


def locate_earliest_divergence(
    predicted: tuple[PrefixDenotation, ...],
    gold: tuple[PrefixDenotation, ...],
    alignment: DenotationAlignment,
) -> ExecutionResidual | None:
    """Return the first certified prefix that lacks an exact aligned denotation."""

    exact_gold_positions = {pair.gold_position for pair in alignment.pairs if pair.exact_match}
    gold_position = next(
        (position for position in range(len(gold)) if position not in exact_gold_positions),
        None,
    )
    if gold_position is None:
        return None
    gold_prefix = gold[gold_position]
    predicted_position = _predicted_candidate_position(
        gold_position,
        alignment,
        len(predicted),
    )
    predicted_prefix = predicted[predicted_position] if predicted_position is not None else None
    similarity = (
        _denotation_similarity(predicted_prefix.denotation, gold_prefix.denotation)[0]
        if predicted_prefix is not None
        else 0.0
    )
    return ExecutionResidual(
        reason=("denotation_mismatch" if predicted_prefix is not None else "missing_prediction"),
        gold_step_index=gold_prefix.step_index,
        predicted_step_index=(
            predicted_prefix.step_index if predicted_prefix is not None else None
        ),
        gold_op=gold_prefix.fields.op,
        predicted_op=(predicted_prefix.fields.op if predicted_prefix is not None else None),
        gold_operation=gold_prefix.operation,
        predicted_operation=(predicted_prefix.operation if predicted_prefix is not None else None),
        gold_fields=gold_prefix.fields,
        predicted_fields=(predicted_prefix.fields if predicted_prefix is not None else None),
        gold_denotation=gold_prefix.denotation,
        predicted_denotation=(
            predicted_prefix.denotation if predicted_prefix is not None else None
        ),
        denotation_similarity=similarity,
    )


def analyze_execution_residual(
    predicted_graphscript: str,
    reference_program_json: str,
    *,
    backend: GraphBackend,
    budget: ExecutionBudget,
    allowed_relations: frozenset[str],
    seed_entity: str | None,
    seed: int,
    trace_id: str,
) -> ResidualAnalysis:
    """Execute, align, and localize a candidate against one certified KQAPro Program."""

    try:
        predicted_script = parse_graphscript(
            predicted_graphscript,
            max_follow_limit=budget.max_follow_limit,
        )
    except GraphScriptError as exc:
        raise ResidualError(exc.reason_code, exc.detail, stage="predicted_parse") from exc
    if predicted_script.version != budget.graphscript_version:
        raise ResidualError(
            "VERSION_MISMATCH",
            f"expected {budget.graphscript_version}, got {predicted_script.version}",
            stage="predicted_parse",
        )
    gold_script = parse_reference_graphscript(
        reference_program_json,
        backend=backend,
        budget=budget,
    )

    predicted_terminal = _execute_terminal(
        predicted_script,
        backend=backend,
        budget=budget,
        seed_entity=seed_entity,
        allowed_relations=allowed_relations,
        trace_id=f"{trace_id}:seed:{seed}:predicted:terminal",
        stage="predicted_execution",
    )
    gold_terminal = _execute_terminal(
        gold_script,
        backend=backend,
        budget=budget,
        seed_entity=seed_entity,
        allowed_relations=allowed_relations,
        trace_id=f"{trace_id}:seed:{seed}:gold:terminal",
        stage="reference_execution",
    )
    predicted_prefixes = execute_output_prefixes(
        predicted_script,
        backend=backend,
        budget=budget,
        seed_entity=seed_entity,
        allowed_relations=allowed_relations,
        trace_id=f"{trace_id}:seed:{seed}:predicted",
    )
    gold_prefixes = execute_output_prefixes(
        gold_script,
        backend=backend,
        budget=budget,
        seed_entity=seed_entity,
        allowed_relations=allowed_relations,
        trace_id=f"{trace_id}:seed:{seed}:gold",
    )
    alignment = align_prefix_denotations(predicted_prefixes, gold_prefixes)
    terminal_metrics = answer_metrics(predicted_terminal.answers, gold_terminal.answers)
    return ResidualAnalysis(
        seed=seed,
        trace_id=trace_id,
        budget=budget,
        predicted_graphscript=_script_payload(predicted_script),
        gold_graphscript=_script_payload(gold_script),
        predicted_prefixes=predicted_prefixes,
        gold_prefixes=gold_prefixes,
        alignment=alignment,
        earliest_divergence=locate_earliest_divergence(
            predicted_prefixes,
            gold_prefixes,
            alignment,
        ),
        predicted_terminal=predicted_terminal,
        gold_terminal=gold_terminal,
        terminal_answer_f1=float(terminal_metrics["f1"]),
        terminal_exact_match=float(terminal_metrics["exact_match"]),
    )
