from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.evaluation import KILTQAMetrics, kilt_qa_metrics
from graphtask_r1.graphscript import (
    EvidenceEpisode,
    EvidenceRetriever,
    EvidenceTurn,
    evidence_solver_prompt,
)
from graphtask_r1.schema import AnswerSet, BenchmarkExample, EvidenceProvenance, PassageHit
from graphtask_r1.training.ms_swift_data import tool_schemas


class EvidenceAgentToolCall(BaseModel):
    """One ordered Hermes tool call within a policy generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: Literal["text_search", "expand_evidence", "select_evidence"]
    arguments: dict[str, Any] = Field(default_factory=dict)


class EvidenceAgentDecision(BaseModel):
    """One policy decision in the same semantic protocol used by ms-swift."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool", "answer"]
    tool_name: Literal["text_search", "expand_evidence", "select_evidence"] | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_calls: tuple[EvidenceAgentToolCall, ...] = ()
    answer: str | None = None


class InteractiveEvidencePolicy(Protocol):
    def decide(
        self,
        messages: tuple[dict[str, Any], ...],
        tools: tuple[dict[str, object], ...],
        *,
        trace_id: str,
    ) -> EvidenceAgentDecision: ...


class InteractiveEvidenceRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    example_id: str
    episode: EvidenceEpisode
    metrics: KILTQAMetrics
    early_answer_attempts: int = Field(ge=0)
    rejection_reason: str | None = None


class InteractiveEvidenceReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    examples: int = Field(ge=1)
    summary: dict[str, float]
    residuals: dict[str, int]
    per_example: tuple[InteractiveEvidenceRun, ...]


def run_interactive_evidence_policy(
    example: BenchmarkExample,
    *,
    retriever: EvidenceRetriever,
    policy: InteractiveEvidencePolicy,
    max_turns: int = 4,
    causal_selection: bool = False,
) -> InteractiveEvidenceRun:
    """Execute the trained retrieve/expand/select/answer protocol at evaluation."""

    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    messages: list[dict[str, Any]] = list(evidence_solver_prompt(example.question))
    tools = tuple(tool_schemas("evidence_solver", text_search_enabled=True))
    observed: dict[str, PassageHit] = {}
    selected: tuple[EvidenceProvenance, ...] = ()
    turns: list[EvidenceTurn] = []
    final_answer = AnswerSet()
    early_answers = 0
    rejection: str | None = None
    for index in range(max_turns):
        try:
            decision = policy.decide(
                tuple(messages),
                tools,
                trace_id=f"interactive:{example.example_id}:{index}",
            )
        except ValueError as exc:
            rejection = str(getattr(exc, "reason_code", "POLICY_OUTPUT_INVALID"))
            break
        if decision.kind == "answer":
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"answer": decision.answer},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
            protocol_complete = _protocol_complete(
                turns,
                causal_selection=causal_selection,
            )
            if not selected or not decision.answer or not protocol_complete:
                early_answers += 1
                messages.append(
                    {
                        "role": "user",
                        "content": _next_protocol_instruction(
                            turns,
                            causal_selection=causal_selection,
                        ),
                    }
                )
                continue
            final_answer = AnswerSet.literals([decision.answer])
            turns.append(
                EvidenceTurn(
                    index=len(turns),
                    action={"op": "answer", "value": decision.answer},
                    selected_evidence=selected,
                )
            )
            break
        if decision.tool_name is None:
            rejection = "INVALID_TOOL_CALL"
            break
        calls = decision.tool_calls or (
            EvidenceAgentToolCall(
                tool_name=decision.tool_name,
                arguments=decision.arguments,
            ),
        )
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in calls
                ],
            }
        )
        for call in calls:
            call_decision = EvidenceAgentDecision(
                kind="tool",
                tool_name=call.tool_name,
                arguments=call.arguments,
            )
            try:
                turn, observation, selected_update = _execute_decision(
                    call_decision,
                    observed=observed,
                    retriever=retriever,
                    trace_id=f"interactive:{example.example_id}:{index}",
                    turn_index=len(turns),
                    prior_turns=turns,
                    causal_selection=causal_selection,
                    question=example.question,
                )
            except (KeyError, TypeError, ValueError) as exc:
                rejection = "INVALID_TOOL_CALL"
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {"error": {"reason_code": rejection, "message": str(exc)}},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
                continue
            turns.append(turn)
            if selected_update is not None:
                selected = selected_update
            messages.append({"role": "tool", "content": observation})
    done = bool(final_answer.answers and selected)
    if not done and rejection is None:
        rejection = "TURN_BUDGET_EXCEEDED"
    episode = EvidenceEpisode(
        question=example.question,
        turns=tuple(turns),
        final_answer=final_answer,
        final_evidence=selected if done else (),
        done=done,
    )
    aliases = example.answer_aliases or tuple(
        (str(answer.value),) for answer in example.gold_answers.answers
    )
    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        aliases,
        example.gold_provenance,
    )
    return InteractiveEvidenceRun(
        example_id=example.example_id,
        episode=episode,
        metrics=metrics,
        early_answer_attempts=early_answers,
        rejection_reason=None if done else rejection,
    )


def evaluate_interactive_evidence(
    examples: Sequence[BenchmarkExample],
    *,
    retriever: EvidenceRetriever,
    policy: InteractiveEvidencePolicy,
    max_turns: int = 4,
    causal_selection: bool = False,
) -> InteractiveEvidenceReport:
    if not examples:
        raise ValueError("at least one evaluation example is required")
    rows = tuple(
        run_interactive_evidence_policy(
            example,
            retriever=retriever,
            policy=policy,
            max_turns=max_turns,
            causal_selection=causal_selection,
        )
        for example in examples
    )
    summary = {
        name: sum(float(getattr(row.metrics, name)) for row in rows) / len(rows)
        for name in KILTQAMetrics.model_fields
    }
    residuals: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.episode.done:
            residuals["joint_success" if row.metrics.kilt_f1 > 0.0 else "joint_failure"] += 1
        else:
            residuals[row.rejection_reason or "incomplete"] += 1
    return InteractiveEvidenceReport(
        examples=len(rows),
        summary=summary,
        residuals=dict(sorted(residuals.items())),
        per_example=rows,
    )


def _execute_decision(
    decision: EvidenceAgentDecision,
    *,
    observed: dict[str, PassageHit],
    retriever: EvidenceRetriever,
    trace_id: str,
    turn_index: int,
    prior_turns: Sequence[EvidenceTurn],
    causal_selection: bool,
    question: str,
) -> tuple[EvidenceTurn, str, tuple[EvidenceProvenance, ...] | None]:
    name = decision.tool_name
    if name is None:
        raise ValueError("tool decision requires tool_name")
    arguments = decision.arguments
    passages: tuple[PassageHit, ...] = ()
    selected_update: tuple[EvidenceProvenance, ...] | None = None
    if name == "text_search":
        passages = retriever.retrieve(
            _required_query(arguments),
            limit=_bounded_limit(arguments),
            trace_id=trace_id,
        )
    elif name == "expand_evidence":
        if not observed:
            raise ValueError("expand_evidence requires prior retrieved passages")
        passages = retriever.expand(
            tuple(observed.values()),
            (
                f"{question} {_required_query(arguments)}"
                if causal_selection
                else _required_query(arguments)
            ),
            limit=_bounded_limit(arguments),
            trace_id=trace_id,
        )
    else:
        raw_keys = arguments.get("passage_keys")
        if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, str | bytes):
            raise ValueError("passage_keys must be a list")
        keys = tuple(dict.fromkeys(str(value) for value in raw_keys))
        if causal_selection:
            retrieve_keys = _turn_passage_keys(prior_turns, "retrieve")
            expand_keys = _turn_passage_keys(prior_turns, "expand")
            if not expand_keys:
                raise ValueError("select_evidence must causally follow expand_evidence")
            if not set(keys) & retrieve_keys or not set(keys) & expand_keys:
                raise ValueError(
                    "selection must cover passages from both retrieve and expand"
                )
        selected_update = tuple(
            EvidenceProvenance(
                page_id=observed[key].page_id,
                paragraph_id=observed[key].paragraph_id,
                title=observed[key].title,
            )
            for key in keys
            if key in observed
        )
        if not selected_update:
            raise ValueError("no selected passage key was observed")
    for passage in passages:
        observed[f"{passage.page_id}:{passage.paragraph_id}"] = passage
    payload = (
        {"selected_passage_keys": [value.passage_key for value in selected_update]}
        if selected_update is not None
        else [_passage_payload(value) for value in passages]
    )
    turn = EvidenceTurn(
        index=turn_index,
        action={"op": _operation_name(name), **arguments},
        passages=passages,
        selected_evidence=selected_update or (),
    )
    return (
        turn,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        selected_update,
    )


def _next_protocol_instruction(
    turns: Sequence[EvidenceTurn], *, causal_selection: bool
) -> str:
    operations = {str(turn.action.get("op")) for turn in turns}
    if "retrieve" not in operations:
        return "Evidence protocol incomplete. Call text_search before answering."
    if "expand" not in operations:
        return "Evidence protocol incomplete. Call expand_evidence before answering."
    if causal_selection and not _protocol_complete(turns, causal_selection=True):
        return (
            "Evidence protocol incomplete. After expand_evidence, call select_evidence "
            "with passage_key values from both retrieve and expand results."
        )
    return "Evidence protocol incomplete. Call select_evidence before answering."


def _turn_passage_keys(turns: Sequence[EvidenceTurn], operation: str) -> set[str]:
    return {
        f"{passage.page_id}:{passage.paragraph_id}"
        for turn in turns
        if str(turn.action.get("op")) == operation
        for passage in turn.passages
    }


def _protocol_complete(
    turns: Sequence[EvidenceTurn], *, causal_selection: bool
) -> bool:
    operations = [str(turn.action.get("op")) for turn in turns]
    if not {"retrieve", "expand", "select_evidence"} <= set(operations):
        return False
    if not causal_selection:
        return True
    expand_index = max(index for index, op in enumerate(operations) if op == "expand")
    select_index = max(
        index for index, op in enumerate(operations) if op == "select_evidence"
    )
    if select_index < expand_index:
        return False
    selected = {
        evidence.passage_key
        for evidence in turns[select_index].selected_evidence
    }
    return bool(
        selected & _turn_passage_keys(turns, "retrieve")
        and selected & _turn_passage_keys(turns, "expand")
    )


def _required_query(arguments: Mapping[str, Any]) -> str:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return query


def _bounded_limit(arguments: Mapping[str, Any]) -> int:
    value = arguments.get("limit", 3)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("limit must be an integer")
    return min(max(int(value), 1), 10)


def _operation_name(
    tool_name: Literal["text_search", "expand_evidence", "select_evidence"],
) -> str:
    return {
        "text_search": "retrieve",
        "expand_evidence": "expand",
        "select_evidence": "select_evidence",
    }[tool_name]


def _passage_payload(passage: PassageHit) -> dict[str, Any]:
    payload = passage.model_dump(mode="json")
    payload["passage_key"] = f"{passage.page_id}:{passage.paragraph_id}"
    return payload
