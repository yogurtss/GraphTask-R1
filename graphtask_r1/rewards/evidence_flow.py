from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.evaluation import kilt_qa_metrics
from graphtask_r1.graphscript.evidence_flow import EvidenceEpisode, ProofExecution
from graphtask_r1.schema import RewardBreakdown

EvidenceResidualKind = Literal["retrieval", "reasoning", "success", "incomplete"]


class EvidenceResidual(BaseModel):
    """A localized learning signal aligned to the certified proof."""

    model_config = ConfigDict(frozen=True)

    kind: EvidenceResidualKind
    recovered_prefix: int = Field(ge=0)
    proof_length: int = Field(ge=1)
    missing_passage_keys: tuple[str, ...] = ()
    answer_f1: float = Field(ge=0.0, le=1.0)

    @property
    def prefix_fraction(self) -> float:
        return self.recovered_prefix / self.proof_length


class EvidencePotentialTrace(BaseModel):
    """Proof-relative progress for a replayable search trajectory.

    ``area`` is the mean potential after every observable action.  It rewards
    recovering certified evidence early while retaining the individual values
    and deltas for audits and future action-level credit assignment.
    """

    model_config = ConfigDict(frozen=True)

    potentials: tuple[float, ...]
    deltas: tuple[float, ...]
    area: float = Field(ge=0.0, le=1.0)
    final: float = Field(ge=0.0, le=1.0)


def classify_evidence_residual(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    answer_aliases: tuple[tuple[str, ...], ...],
) -> EvidenceResidual:
    """Separate evidence acquisition failures from answer reasoning failures."""

    retrieved = {value.passage_key for value in episode.retrieved_provenance()}
    prefix = 0
    for passage_key in proof.passage_keys:
        if passage_key not in retrieved:
            break
        prefix += 1
    missing = tuple(key for key in proof.passage_keys if key not in retrieved)
    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        answer_aliases,
        (proof.evidence,),
    )
    if not episode.done:
        kind: EvidenceResidualKind = "incomplete"
    elif missing:
        kind = "retrieval"
    elif metrics.answer_f1 < 1.0:
        kind = "reasoning"
    else:
        kind = "success"
    return EvidenceResidual(
        kind=kind,
        recovered_prefix=prefix,
        proof_length=len(proof.passage_keys),
        missing_passage_keys=missing,
        answer_f1=metrics.answer_f1,
    )


def evidence_frontier_reward(
    evidence_recovery_rate: float,
    conditional_answer_rate: float,
    *,
    target: float = 0.5,
    sigma: float = 0.2,
) -> float:
    """Prefer tasks near both the retriever and reasoner competence frontiers."""

    for value in (evidence_recovery_rate, conditional_answer_rate, target):
        if not 0.0 <= value <= 1.0:
            raise ValueError("frontier rates must be in [0, 1]")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    evidence_term = (evidence_recovery_rate - target) ** 2
    reasoning_term = (conditional_answer_rate - target) ** 2
    return math.exp(-(evidence_term + reasoning_term) / (2 * sigma**2))


def evidence_questioner_reward(
    evidence_recovery_rate: float,
    conditional_answer_rate: float,
    *,
    novelty: float = 1.0,
    target: float = 0.5,
    sigma: float = 0.2,
) -> RewardBreakdown:
    """Reward proof-valid tasks near both Solver competence boundaries."""

    if not 0.0 <= novelty <= 1.0:
        raise ValueError("novelty must be in [0, 1]")
    frontier = evidence_frontier_reward(
        evidence_recovery_rate,
        conditional_answer_rate,
        target=target,
        sigma=sigma,
    )
    components = {
        "proof_validity": 0.20,
        "dual_frontier": 0.65 * frontier,
        "novelty": 0.15 * novelty,
    }
    return RewardBreakdown(
        total=sum(components.values()),
        components=components,
        metadata={
            "evidence_recovery_rate": evidence_recovery_rate,
            "conditional_answer_rate": conditional_answer_rate,
        },
    )


def evidence_solver_reward(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    answer_aliases: tuple[tuple[str, ...], ...],
) -> RewardBreakdown:
    """Reward exact evidence-grounded answers and retain the two residual signals."""

    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        answer_aliases,
        (proof.evidence,),
    )
    residual = classify_evidence_residual(proof, episode, answer_aliases)
    search_turns = sum(turn.action.get("op") in {"retrieve", "expand"} for turn in episode.turns)
    efficiency = 1.0 / max(1, search_turns)
    components = {
        "answer_f1": 0.30 * metrics.answer_f1,
        "kilt_f1": 0.25 * metrics.kilt_f1,
        "provenance_rprecision": 0.20 * metrics.provenance_rprecision,
        "proof_prefix": 0.15 * residual.prefix_fraction,
        "grounded_efficiency": 0.10 * efficiency * metrics.kilt_exact_match,
    }
    return RewardBreakdown(
        total=sum(components.values()),
        components=components,
        metadata={
            "residual_kind": residual.kind,
            "missing_passage_keys": list(residual.missing_passage_keys),
        },
    )


def evidence_search_r1_em_reward(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    answer_aliases: tuple[tuple[str, ...], ...],
) -> RewardBreakdown:
    """Protocol-matched Search-R1 baseline using final-answer EM only.

    Search-R1 deliberately omits format and process rewards.  ``proof`` is used
    only to execute the certified gold answer and is not exposed to the reward
    beyond the answer aliases derived from that execution.
    """

    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        answer_aliases,
        (proof.evidence,),
    )
    exact_match = metrics.answer_exact_match
    return RewardBreakdown(
        total=exact_match,
        components={"answer_exact_match": exact_match},
        metadata={"reward_variant": "search_r1_em"},
    )


def evidence_proof_potential_trace(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    *,
    prefix_weight: float = 0.7,
) -> EvidencePotentialTrace:
    """Measure ordered proof recovery after each evidence action.

    The prefix term distinguishes finding the bridge from finding a later hop
    out of order.  The set-F1 term still gives useful partial credit when a
    valid proof passage is recovered without completing the prefix.
    """

    if not 0.0 <= prefix_weight <= 1.0:
        raise ValueError("prefix_weight must be in [0, 1]")
    proof_keys = tuple(dict.fromkeys(proof.passage_keys))
    observed: set[str] = set()
    potentials: list[float] = []
    for turn in episode.turns:
        observed.update(
            f"{passage.page_id}:{passage.paragraph_id}" for passage in turn.passages
        )
        selected = {value.passage_key for value in turn.selected_evidence}
        observed.update(selected)
        potentials.append(
            _proof_potential(
                proof_keys,
                selected or observed,
                prefix_weight=prefix_weight,
            )
        )
    deltas: list[float] = []
    previous = 0.0
    for value in potentials:
        deltas.append(value - previous)
        previous = value
    final = potentials[-1] if potentials else 0.0
    area = sum(potentials) / len(potentials) if potentials else 0.0
    return EvidencePotentialTrace(
        potentials=tuple(potentials),
        deltas=tuple(deltas),
        area=area,
        final=final,
    )


def evidence_ecp_solver_reward(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    answer_aliases: tuple[tuple[str, ...], ...],
) -> RewardBreakdown:
    """Joint-first reward for executable counterfactual proof self-play.

    Process-only credit is capped at 0.30, whereas a fully grounded answer
    receives at least 0.70.  This prevents answer and provenance components
    from compensating for one another as they can in the legacy additive
    reward, while failed trajectories still provide bounded learning signal.
    """

    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        answer_aliases,
        (proof.evidence,),
    )
    trace = evidence_proof_potential_trace(proof, episode)
    selected_f1 = _set_f1(
        {value.passage_key for value in episode.final_evidence},
        set(proof.passage_keys),
    )
    components = {
        "joint_outcome": 0.70 * metrics.kilt_f1,
        "proof_progress_auc": 0.20 * trace.area,
        "selected_proof_f1": 0.10 * selected_f1,
    }
    residual = classify_evidence_residual(proof, episode, answer_aliases)
    return RewardBreakdown(
        total=sum(components.values()),
        components=components,
        metadata={
            "reward_variant": "ecp_v1",
            "residual_kind": residual.kind,
            "proof_potentials": list(trace.potentials),
            "proof_potential_deltas": list(trace.deltas),
            "missing_passage_keys": list(residual.missing_passage_keys),
        },
    )


def evidence_ecp_answer_gated_reward(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    answer_aliases: tuple[tuple[str, ...], ...],
) -> RewardBreakdown:
    """Gate every proof-shaping term by executable answer correctness.

    ECP-v2 removes the process-only optimum exposed by v1: a trajectory cannot
    earn proof or provenance credit unless it both completes the tool protocol
    and answers correctly. Among correct answers, KILT joint quality and early
    proof recovery still determine the preference ordering used by GRPO.
    """

    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        answer_aliases,
        (proof.evidence,),
    )
    trace = evidence_proof_potential_trace(proof, episode)
    selected_f1 = _set_f1(
        {value.passage_key for value in episode.final_evidence},
        set(proof.passage_keys),
    )
    answer_gate = metrics.answer_f1 * float(episode.done)
    components = {
        "answer_credit": 0.55 * answer_gate,
        "joint_evidence": 0.30 * metrics.kilt_f1 * float(episode.done),
        "answer_gated_proof_auc": 0.10 * answer_gate * trace.area,
        "answer_gated_selection": 0.05 * answer_gate * selected_f1,
    }
    residual = classify_evidence_residual(proof, episode, answer_aliases)
    return RewardBreakdown(
        total=sum(components.values()),
        components=components,
        metadata={
            "reward_variant": "ecp_v2",
            "residual_kind": residual.kind,
            "answer_gate": answer_gate,
            "proof_potentials": list(trace.potentials),
            "proof_potential_deltas": list(trace.deltas),
            "missing_passage_keys": list(residual.missing_passage_keys),
        },
    )


def evidence_ecp_protocol_gated_reward(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    answer_aliases: tuple[tuple[str, ...], ...],
) -> RewardBreakdown:
    """Open proof credit only after exact answer and protocol certification.

    The gate is deliberately exact rather than token-overlap based: bridge and
    target titles often share tokens, so an F1 gate can reward copying the
    bridge.  This version makes the executable proof a causal prerequisite for
    credit while retaining dense preferences among certified successes.
    """

    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        answer_aliases,
        (proof.evidence,),
    )
    trace = evidence_proof_potential_trace(proof, episode)
    selected_f1 = _set_f1(
        {value.passage_key for value in episode.final_evidence},
        set(proof.passage_keys),
    )
    operations = {str(turn.action.get("op")) for turn in episode.turns}
    protocol_complete = {"retrieve", "expand", "select_evidence"} <= operations
    answer_gate = metrics.answer_exact_match * float(episode.done and protocol_complete)
    components = {
        "exact_answer_credit": 0.50 * answer_gate,
        "joint_evidence": 0.35 * metrics.kilt_f1 * answer_gate,
        "exact_gated_proof_auc": 0.10 * answer_gate * trace.area,
        "exact_gated_selection": 0.05 * answer_gate * selected_f1,
    }
    residual = classify_evidence_residual(proof, episode, answer_aliases)
    return RewardBreakdown(
        total=sum(components.values()),
        components=components,
        metadata={
            "reward_variant": "ecp_v3",
            "residual_kind": residual.kind,
            "answer_gate": answer_gate,
            "protocol_complete": protocol_complete,
            "proof_potentials": list(trace.potentials),
            "proof_potential_deltas": list(trace.deltas),
            "missing_passage_keys": list(residual.missing_passage_keys),
        },
    )


def evidence_ecp_causal_reward(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    answer_aliases: tuple[tuple[str, ...], ...],
) -> RewardBreakdown:
    """Reward a complete proof selected causally after hyperlink expansion.

    The proof certificate is binary: selecting only the easy bridge earns no
    process credit. It can provide a bounded subgoal reward before the answer is
    correct, while answer and joint KILT terms remain non-compensatory.
    """

    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        answer_aliases,
        (proof.evidence,),
    )
    trace = evidence_proof_potential_trace(proof, episode)
    operations = [str(turn.action.get("op")) for turn in episode.turns]
    expand_indices = [index for index, op in enumerate(operations) if op == "expand"]
    select_indices = [
        index for index, op in enumerate(operations) if op == "select_evidence"
    ]
    causal_order = bool(
        expand_indices and select_indices and select_indices[-1] > expand_indices[-1]
    )
    selected_keys = {value.passage_key for value in episode.final_evidence}
    proof_complete = set(proof.passage_keys) <= selected_keys
    certificate = float(causal_order and proof_complete)
    exact_answer = metrics.answer_exact_match * float(episode.done)
    components = {
        "exact_answer_credit": 0.45 * exact_answer,
        "causal_proof_certificate": 0.25 * certificate,
        "joint_evidence": 0.25 * metrics.kilt_f1 * certificate,
        "certificate_gated_proof_auc": 0.05 * certificate * trace.area,
    }
    residual = classify_evidence_residual(proof, episode, answer_aliases)
    return RewardBreakdown(
        total=sum(components.values()),
        components=components,
        metadata={
            "reward_variant": "ecp_v4",
            "residual_kind": residual.kind,
            "causal_order": causal_order,
            "proof_complete": proof_complete,
            "proof_potentials": list(trace.potentials),
            "missing_passage_keys": list(residual.missing_passage_keys),
        },
    )


def evidence_causal_curriculum_trace(
    proof: ProofExecution,
    episode: EvidenceEpisode,
) -> EvidencePotentialTrace:
    """Track KQAPro-style grounding stages for an open-text proof.

    The potential is monotone and milestone based: recover the bridge with
    retrieval, recover later proof hops with expansion, then select the proof
    after expansion. Repeated calls cannot accumulate additional credit.
    """

    proof_keys = tuple(dict.fromkeys(proof.passage_keys))
    bridge_key = proof_keys[0]
    later_keys = set(proof_keys[1:])
    proof_set = set(proof_keys)
    retrieved: set[str] = set()
    expanded: set[str] = set()
    latest_expand = -1
    previous = 0.0
    potentials: list[float] = []
    deltas: list[float] = []
    for index, turn in enumerate(episode.turns):
        operation = str(turn.action.get("op"))
        passage_keys = {
            f"{passage.page_id}:{passage.paragraph_id}" for passage in turn.passages
        }
        if operation == "retrieve":
            retrieved.update(passage_keys)
        elif operation == "expand":
            expanded.update(passage_keys)
            latest_expand = index

        bridge_grounded = float(bridge_key in retrieved)
        expansion_fraction = (
            len(expanded & later_keys) / len(later_keys)
            if bridge_grounded and later_keys
            else bridge_grounded
        )
        raw_selected = turn.action.get("passage_keys", [])
        selected = (
            {str(value) for value in raw_selected}
            if isinstance(raw_selected, Sequence)
            and not isinstance(raw_selected, str | bytes)
            else set()
        )
        selected.update(value.passage_key for value in turn.selected_evidence)
        causal_selection = bool(
            operation == "select_evidence"
            and latest_expand >= 0
            and index > latest_expand
            and bridge_key in selected
            and (not later_keys or bool(selected & later_keys))
        )
        selection_fraction = (
            len(selected & proof_set) / len(proof_set) if causal_selection else 0.0
        )
        potential = max(
            previous,
            0.20 * bridge_grounded
            + 0.35 * expansion_fraction
            + 0.45 * selection_fraction,
        )
        potentials.append(potential)
        deltas.append(potential - previous)
        previous = potential
    return EvidencePotentialTrace(
        potentials=tuple(potentials),
        deltas=tuple(deltas),
        area=sum(potentials) / len(potentials) if potentials else 0.0,
        final=potentials[-1] if potentials else 0.0,
    )


def evidence_ecp_curriculum_reward(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    answer_aliases: tuple[tuple[str, ...], ...],
) -> RewardBreakdown:
    """Combine staged executable grounding with a terminal causal certificate.

    This transfers KQAPro's syntax/grounding/solve curriculum to KILT without
    importing KQAPro examples or weights. Process shaping is capped at 0.15;
    answer and joint evidence remain the dominant objective.
    """

    metrics = kilt_qa_metrics(
        episode.final_answer,
        episode.final_evidence,
        answer_aliases,
        (proof.evidence,),
    )
    trace = evidence_causal_curriculum_trace(proof, episode)
    certificate = float(math.isclose(trace.final, 1.0))
    exact_answer = metrics.answer_exact_match * float(episode.done)
    components = {
        "curriculum_potential": 0.15 * trace.final,
        "causal_proof_certificate": 0.15 * certificate,
        "exact_answer_credit": 0.40 * exact_answer,
        "joint_evidence": 0.30 * metrics.kilt_f1 * certificate,
    }
    residual = classify_evidence_residual(proof, episode, answer_aliases)
    return RewardBreakdown(
        total=sum(components.values()),
        components=components,
        metadata={
            "reward_variant": "ecp_v5",
            "residual_kind": residual.kind,
            "causal_certificate": bool(certificate),
            "curriculum_potentials": list(trace.potentials),
            "curriculum_deltas": list(trace.deltas),
            "missing_passage_keys": list(residual.missing_passage_keys),
        },
    )


def _proof_potential(
    proof_keys: Sequence[str],
    observed: set[str],
    *,
    prefix_weight: float,
) -> float:
    if not proof_keys:
        return 0.0
    prefix = 0
    for key in proof_keys:
        if key not in observed:
            break
        prefix += 1
    prefix_fraction = prefix / len(proof_keys)
    return prefix_weight * prefix_fraction + (1.0 - prefix_weight) * _set_f1(
        observed,
        set(proof_keys),
    )


def _set_f1(predicted: set[str], reference: set[str]) -> float:
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    overlap = len(predicted & reference)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2.0 * precision * recall / (precision + recall)
