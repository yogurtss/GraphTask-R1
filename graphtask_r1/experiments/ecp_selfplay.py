from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.evaluation import normalize_openqa_answer
from graphtask_r1.experiments.evidence_selfplay import EvidenceChallenge
from graphtask_r1.graph import GraphBackend
from graphtask_r1.graphscript import (
    CounterfactualRerankerState,
    ProofExecution,
    ProofScript,
    execute_proofscript,
)
from graphtask_r1.graphscript.evidence_flow import PassageAddressBackend
from graphtask_r1.schema import PassageHit


class CounterfactualProofError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class CounterfactualProofPair(BaseModel):
    """Two executable proofs separated by one causal graph intervention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "ecp-pair-v1"
    bridge_page_id: str
    positive_challenge_id: str
    counterfactual_challenge_id: str
    positive_question: str
    counterfactual_question: str
    positive_proof: ProofScript
    counterfactual_proof: ProofScript
    positive_execution: ProofExecution
    counterfactual_execution: ProofExecution
    changed_operation_index: int = Field(ge=0)
    shared_passage_keys: tuple[str, ...] = Field(min_length=1)
    positive_only_passage_keys: tuple[str, ...] = Field(min_length=1)
    counterfactual_only_passage_keys: tuple[str, ...] = Field(min_length=1)


class CounterfactualRetrievalExample(BaseModel):
    """Contrastive retriever supervision derived without an LLM judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "ecp-retrieval-v1"
    pair_id: str
    query: str
    bridge_page_id: str
    positive_passage: PassageHit
    hard_negative_passages: tuple[PassageHit, ...] = Field(min_length=1)


def certify_counterfactual_pair(
    positive: EvidenceChallenge,
    counterfactual: EvidenceChallenge,
    *,
    backend: GraphBackend,
) -> CounterfactualProofPair:
    """Certify a minimal target-edge intervention by executing both proofs."""

    if positive.bridge_page_id != counterfactual.bridge_page_id:
        raise CounterfactualProofError(
            "BRIDGE_MISMATCH",
            "counterfactual proofs must start from the same bridge page",
        )
    if positive.target_page_id == counterfactual.target_page_id:
        raise CounterfactualProofError(
            "TARGET_UNCHANGED",
            "counterfactual target must differ from the positive target",
        )
    changed = _changed_operations(positive.proof, counterfactual.proof)
    if len(changed) != 1:
        raise CounterfactualProofError(
            "NON_MINIMAL_INTERVENTION",
            f"expected one changed proof operation, found {len(changed)}",
        )
    changed_index = changed[0]
    positive_op = positive.proof.ops[changed_index].model_dump(mode="json", by_alias=True)
    negative_op = counterfactual.proof.ops[changed_index].model_dump(
        mode="json", by_alias=True
    )
    if (
        positive_op.get("op") != "follow_anchor"
        or negative_op.get("op") != "follow_anchor"
        or _different_fields(positive_op, negative_op) != {"target_page_id"}
    ):
        raise CounterfactualProofError(
            "INVALID_CAUSAL_INTERVENTION",
            "the sole change must replace one follow_anchor target_page_id",
        )

    positive_execution = execute_proofscript(positive.proof, backend)
    counterfactual_execution = execute_proofscript(counterfactual.proof, backend)
    positive_answers = {
        normalize_openqa_answer(str(value.value))
        for value in positive_execution.answers.answers
    }
    counterfactual_answers = {
        normalize_openqa_answer(str(value.value))
        for value in counterfactual_execution.answers.answers
    }
    if positive_answers & counterfactual_answers:
        raise CounterfactualProofError(
            "ANSWER_UNCHANGED",
            "the minimal graph intervention did not change the executed answer",
        )

    positive_keys = set(positive_execution.passage_keys)
    counterfactual_keys = set(counterfactual_execution.passage_keys)
    shared = tuple(key for key in positive_execution.passage_keys if key in counterfactual_keys)
    positive_only = tuple(
        key for key in positive_execution.passage_keys if key not in counterfactual_keys
    )
    counterfactual_only = tuple(
        key for key in counterfactual_execution.passage_keys if key not in positive_keys
    )
    if not shared:
        raise CounterfactualProofError(
            "NO_SHARED_CAUSAL_CONTEXT",
            "proofs must retain at least one common bridge passage",
        )
    if not positive_only or not counterfactual_only:
        raise CounterfactualProofError(
            "NO_CONTRASTIVE_EVIDENCE",
            "both proofs must contain evidence unique to their target",
        )
    return CounterfactualProofPair(
        bridge_page_id=positive.bridge_page_id,
        positive_challenge_id=positive.challenge_id,
        counterfactual_challenge_id=counterfactual.challenge_id,
        positive_question=positive.example.question,
        counterfactual_question=counterfactual.example.question,
        positive_proof=positive.proof,
        counterfactual_proof=counterfactual.proof,
        positive_execution=positive_execution,
        counterfactual_execution=counterfactual_execution,
        changed_operation_index=changed_index,
        shared_passage_keys=shared,
        positive_only_passage_keys=positive_only,
        counterfactual_only_passage_keys=counterfactual_only,
    )


def generate_counterfactual_proof_pairs(
    challenges: Sequence[EvidenceChallenge],
    *,
    backend: GraphBackend,
    seed: int,
    limit: int | None = None,
) -> tuple[CounterfactualProofPair, ...]:
    """Build deterministic directional pairs within each private bridge pool."""

    grouped: dict[str, list[EvidenceChallenge]] = defaultdict(list)
    for challenge in challenges:
        grouped[challenge.bridge_page_id].append(challenge)
    candidates: list[tuple[EvidenceChallenge, EvidenceChallenge]] = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda value: value.challenge_id)
        candidates.extend(
            (positive, counterfactual)
            for positive in ordered
            for counterfactual in ordered
            if positive.challenge_id != counterfactual.challenge_id
        )
    random.Random(seed).shuffle(candidates)
    pairs: list[CounterfactualProofPair] = []
    for positive, counterfactual in candidates:
        try:
            pair = certify_counterfactual_pair(
                positive,
                counterfactual,
                backend=backend,
            )
        except CounterfactualProofError:
            continue
        pairs.append(pair)
        if limit is not None and len(pairs) >= limit:
            break
    return tuple(pairs)


def counterfactual_retrieval_examples(
    pairs: Sequence[CounterfactualProofPair],
    *,
    backend: GraphBackend,
) -> tuple[CounterfactualRetrievalExample, ...]:
    """Materialize proof-exclusive positives and hard negatives."""

    if not isinstance(backend, PassageAddressBackend):
        raise CounterfactualProofError(
            "PASSAGE_LOOKUP_UNAVAILABLE",
            "counterfactual retrieval export requires addressable passages",
        )
    rows: list[CounterfactualRetrievalExample] = []
    for pair in pairs:
        negatives = tuple(
            _passage(backend, key) for key in pair.counterfactual_only_passage_keys
        )
        for key in pair.positive_only_passage_keys:
            rows.append(
                CounterfactualRetrievalExample(
                    pair_id=(
                        f"{pair.positive_challenge_id}::"
                        f"{pair.counterfactual_challenge_id}"
                    ),
                    query=pair.positive_question,
                    bridge_page_id=pair.bridge_page_id,
                    positive_passage=_passage(backend, key),
                    hard_negative_passages=negatives,
                )
            )
    return tuple(rows)


def train_counterfactual_reranker(
    examples: Sequence[CounterfactualRetrievalExample],
    *,
    backend: GraphBackend,
    seed: int,
    epochs: int = 25,
    learning_rate: float = 0.25,
    l2: float = 0.001,
) -> CounterfactualRerankerState:
    """Fit a deterministic pairwise reranker from executable interventions.

    Each update increases the score margin between a proof-exclusive positive
    target and an alternative target linked from the same bridge. Features are
    query-token matches against passage text and graph types, so no gold answer
    string or generated rationale is used.
    """

    if not examples:
        raise ValueError("counterfactual reranker requires training examples")
    pairs = [
        (example.query, example.positive_passage, negative)
        for example in examples
        for negative in example.hard_negative_passages
    ]
    rng = random.Random(seed)
    weights: dict[str, float] = {}
    for _ in range(epochs):
        rng.shuffle(pairs)
        for query, positive, negative in pairs:
            query_tokens = _retrieval_tokens(query)
            positive_tokens = _retrieval_document_tokens(positive, backend)
            negative_tokens = _retrieval_document_tokens(negative, backend)
            positive_features = query_tokens & positive_tokens
            negative_features = query_tokens & negative_tokens
            feature_difference = {
                token: float(token in positive_features) - float(token in negative_features)
                for token in positive_features | negative_features
            }
            margin = sum(
                weights.get(token, 0.0) * difference
                for token, difference in feature_difference.items()
            )
            pairwise_gradient = 1.0 / (1.0 + math.exp(min(30.0, margin)))
            for token, difference in feature_difference.items():
                current = weights.get(token, 0.0)
                weights[token] = current * (1.0 - learning_rate * l2) + (
                    learning_rate * pairwise_gradient * difference
                )
    return CounterfactualRerankerState(
        seed=seed,
        examples=len(examples),
        epochs=epochs,
        token_weights={
            token: weight
            for token, weight in sorted(weights.items())
            if abs(weight) >= 1e-8
        },
    )


def _retrieval_document_tokens(passage: PassageHit, backend: GraphBackend) -> frozenset[str]:
    type_text = " ".join(backend.entity_info(passage.page_id).type_ids)
    return _retrieval_tokens(f"{passage.title} {passage.text} {type_text}")


def _retrieval_tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _changed_operations(left: ProofScript, right: ProofScript) -> tuple[int, ...]:
    if len(left.ops) != len(right.ops):
        return tuple(range(max(len(left.ops), len(right.ops))))
    return tuple(
        index
        for index, (left_op, right_op) in enumerate(zip(left.ops, right.ops, strict=True))
        if left_op.model_dump(mode="json", by_alias=True)
        != right_op.model_dump(mode="json", by_alias=True)
    )


def _different_fields(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    return {
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    }


def _passage(backend: PassageAddressBackend, passage_key: str) -> PassageHit:
    page_id, separator, paragraph = passage_key.rpartition(":")
    if not separator or not page_id:
        raise CounterfactualProofError(
            "INVALID_PASSAGE_KEY",
            f"invalid proof passage key: {passage_key}",
        )
    try:
        paragraph_id = int(paragraph)
    except ValueError as exc:
        raise CounterfactualProofError(
            "INVALID_PASSAGE_KEY",
            f"invalid proof passage key: {passage_key}",
        ) from exc
    return PassageHit.model_validate(backend.get_passage(page_id, paragraph_id))
