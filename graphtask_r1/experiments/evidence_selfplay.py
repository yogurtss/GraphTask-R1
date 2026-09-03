from __future__ import annotations

import random
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.evaluation import normalize_openqa_answer
from graphtask_r1.experiments.kilt_evidence_ab import (
    EvidenceABConfig,
    EvidenceAnswerer,
    EvidencePolicyRun,
    HyperlinkEvidenceFlow,
    SingleRetrievalBaseline,
)
from graphtask_r1.experiments.transformers_answerer import evidence_answer_prompt
from graphtask_r1.graph import GraphBackend
from graphtask_r1.graphscript import EvidenceRetriever, ProofExecution, ProofScript
from graphtask_r1.graphscript.evidence_flow import execute_proofscript
from graphtask_r1.rewards import EvidenceResidual, evidence_frontier_reward
from graphtask_r1.rewards.evidence_flow import EvidenceResidualKind
from graphtask_r1.schema import BenchmarkExample
from graphtask_r1.training.sft_dataset import SFT_SCHEMA
from graphtask_r1.utils import (
    ParquetRowWriter,
    read_json,
    stable_hash,
    write_json,
    write_records,
)

_WEAK_CATEGORY_TOKENS = frozenset(
    {"the", "of", "and", "in", "from", "people", "births", "deaths", "articles"}
)


class EvidenceSelfPlayConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    candidate_limit: int = Field(default=32, ge=2)
    train_ratio: float = Field(default=0.75, gt=0.0, lt=1.0)
    retrieve_k: int = Field(default=3, ge=1, le=10)
    expand_k: int = Field(default=3, ge=1, le=10)
    context_k: int = Field(default=5, ge=2, le=20)
    frontier_target: float = Field(default=0.5, ge=0.0, le=1.0)
    frontier_sigma: float = Field(default=0.2, gt=0.0)

    @property
    def ab_config(self) -> EvidenceABConfig:
        return EvidenceABConfig(
            retrieve_k=self.retrieve_k,
            expand_k=self.expand_k,
            context_k=self.context_k,
        )


class EvidenceChallenge(BaseModel):
    model_config = ConfigDict(frozen=True)

    challenge_id: str
    bridge_page_id: str
    target_page_id: str
    category: str
    example: BenchmarkExample
    proof: ProofScript


class EvidenceSelfPlayRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    challenge: EvidenceChallenge
    proof_execution: ProofExecution
    baseline_run: EvidencePolicyRun
    evidence_flow_run: EvidencePolicyRun
    residual: EvidenceResidual
    questioner_reward: float = Field(ge=0.0, le=1.0)


class EvidenceSelfPlayRound(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    records: tuple[EvidenceSelfPlayRecord, ...]
    retrieval_residuals: int = Field(ge=0)
    reasoning_residuals: int = Field(ge=0)
    successes: int = Field(ge=0)


def generate_evidence_challenges(
    backend: GraphBackend,
    config: EvidenceSelfPlayConfig,
) -> tuple[EvidenceChallenge, ...]:
    """Rule Questioner: propose certified bridge/category challenges from KILT links."""

    page_ids = set(backend.all_entities(limit=100_000))
    candidates: list[EvidenceChallenge] = []
    for bridge_id in sorted(page_ids):
        targets = sorted(
            {
                edge.object
                for edge in backend.neighbors(
                    [bridge_id],
                    direction="out",
                    relation_ids=["wikipedia_link"],
                    limit=1_000,
                    trace_id=f"evidence-questioner:{bridge_id}",
                )
                if edge.object in page_ids and edge.object != bridge_id
            }
        )
        if not targets:
            continue
        target_types = {
            target_id: tuple(
                value
                for value in backend.entity_info(target_id).type_ids
                if value.startswith("category:")
            )
            for target_id in targets
        }
        type_frequency = {
            type_id: sum(type_id in values for values in target_types.values())
            for values in target_types.values()
            for type_id in values
        }
        for target_id in targets:
            target_label = backend.entity_info(target_id).label
            categories = sorted(
                (
                    type_id.removeprefix("category:")
                    for type_id in target_types[target_id]
                    if type_frequency[type_id] == 1
                    and not _category_leaks_answer(type_id, target_label)
                ),
                key=lambda value: (len(value), value),
            )
            if not categories:
                continue
            category = categories[0]
            challenge = _challenge(
                backend,
                bridge_id=bridge_id,
                target_id=target_id,
                category=category,
            )
            if challenge is not None:
                candidates.append(challenge)
    random.Random(config.seed).shuffle(candidates)
    return tuple(candidates[: config.candidate_limit])


def run_evidence_selfplay_round(
    challenges: Sequence[EvidenceChallenge],
    *,
    backend: GraphBackend,
    retriever: EvidenceRetriever,
    answerer: EvidenceAnswerer,
    config: EvidenceSelfPlayConfig,
) -> EvidenceSelfPlayRound:
    """Evaluate the frozen Solver and localize each Questioner challenge residual."""

    baseline = SingleRetrievalBaseline(retriever, answerer, config.ab_config)
    evidence_flow = HyperlinkEvidenceFlow(retriever, answerer, config.ab_config)
    records: list[EvidenceSelfPlayRecord] = []
    for challenge in challenges:
        proof = execute_proofscript(challenge.proof, backend)
        baseline_run = baseline.run(challenge.example)
        flow_run = evidence_flow.run(challenge.example)
        residual = _policy_residual(challenge, proof, flow_run)
        recovery = residual.prefix_fraction
        conditional_answer = residual.answer_f1 if recovery == 1.0 else 0.0
        records.append(
            EvidenceSelfPlayRecord(
                challenge=challenge,
                proof_execution=proof,
                baseline_run=baseline_run,
                evidence_flow_run=flow_run,
                residual=residual,
                questioner_reward=evidence_frontier_reward(
                    recovery,
                    conditional_answer,
                    target=config.frontier_target,
                    sigma=config.frontier_sigma,
                ),
            )
        )
    counts = {kind: sum(record.residual.kind == kind for record in records) for kind in (
        "retrieval",
        "reasoning",
        "success",
    )}
    return EvidenceSelfPlayRound(
        seed=config.seed,
        records=tuple(records),
        retrieval_residuals=counts["retrieval"],
        reasoning_residuals=counts["reasoning"],
        successes=counts["success"],
    )


def export_evidence_selfplay_round(
    round_result: EvidenceSelfPlayRound,
    output_dir: Path,
    *,
    config: EvidenceSelfPlayConfig,
    split_manifest: Path | None = None,
) -> dict[str, object]:
    """Freeze validation first, then export reasoning-residual Solver supervision."""

    records = list(round_result.records)
    train_records, val_records, split_payload = _split_records(
        records,
        config=config,
        split_manifest=split_manifest,
    )
    train_rows = [
        _sft_row(record, split="train")
        for record in train_records
        if record.residual.kind == "reasoning"
    ]
    val_rows = [_sft_row(record, split="val") for record in val_records]
    write_records(
        output_dir / "records.parquet",
        (record.model_dump(mode="json") for record in records),
    )
    _write_sft_rows(output_dir / "train.parquet", train_rows)
    _write_sft_rows(output_dir / "val.parquet", val_rows)
    write_json(output_dir / "split.json", split_payload)
    summary: dict[str, object] = {
        "seed": config.seed,
        "challenges": len(records),
        "train_candidates": len(train_records),
        "train_reasoning_residuals": len(train_rows),
        "validation_examples": len(val_rows),
        "retrieval_residuals": round_result.retrieval_residuals,
        "reasoning_residuals": round_result.reasoning_residuals,
        "successes": round_result.successes,
        "train_residual_counts": _residual_counts(train_records),
        "validation_residual_counts": _residual_counts(val_records),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def _split_records(
    records: list[EvidenceSelfPlayRecord],
    *,
    config: EvidenceSelfPlayConfig,
    split_manifest: Path | None,
) -> tuple[
    list[EvidenceSelfPlayRecord],
    list[EvidenceSelfPlayRecord],
    dict[str, object],
]:
    by_id = {record.challenge.challenge_id: record for record in records}
    if split_manifest is not None and split_manifest.exists():
        raw = read_json(split_manifest)
        train_ids = [str(value) for value in raw["train_ids"]]
        val_ids = [str(value) for value in raw["validation_ids"]]
        train = [by_id[value] for value in train_ids if value in by_id]
        val = [by_id[value] for value in val_ids if value in by_id]
    else:
        ranked = sorted(
            records,
            key=lambda record: (
                stable_hash([str(config.seed), record.challenge.challenge_id]),
                record.challenge.challenge_id,
            ),
        )
        train_count = max(
            1,
            min(len(ranked) - 1, round(len(ranked) * config.train_ratio)),
        )
        train, val = ranked[:train_count], ranked[train_count:]
        reasoning_train = next(
            (record for record in train if record.residual.kind == "reasoning"),
            None,
        )
        if reasoning_train is not None and not any(
            record.residual.kind == "reasoning" for record in val
        ):
            replacement = next(
                (record for record in val if record.residual.kind != "reasoning"),
                None,
            )
            if replacement is not None:
                train.remove(reasoning_train)
                val.remove(replacement)
                train.append(replacement)
                val.append(reasoning_train)
    manifest: dict[str, object] = {
        "seed": config.seed,
        "train_ids": [record.challenge.challenge_id for record in train],
        "validation_ids": [record.challenge.challenge_id for record in val],
    }
    return train, val, manifest


def _residual_counts(records: list[EvidenceSelfPlayRecord]) -> dict[str, int]:
    return {
        kind: sum(record.residual.kind == kind for record in records)
        for kind in ("retrieval", "reasoning", "success", "incomplete")
    }


def _challenge(
    backend: GraphBackend,
    *,
    bridge_id: str,
    target_id: str,
    category: str,
) -> EvidenceChallenge | None:
    bridge = backend.entity_info(bridge_id)
    target = backend.entity_info(target_id)
    proof = ProofScript.model_validate(
        {
            "version": "0.4-proof",
            "ops": [
                {"op": "page", "page_id": bridge_id, "out": "h0"},
                {"op": "paragraph", "in": "h0", "paragraph_id": 0, "out": "h1"},
                {
                    "op": "follow_anchor",
                    "in": "h0",
                    "target_page_id": target_id,
                    "out": "h2",
                },
                {"op": "paragraph", "in": "h2", "paragraph_id": 0, "out": "h3"},
                {"op": "page_title", "in": "h2", "out": "h4"},
                {"op": "join_evidence", "inputs": ["h1", "h3"], "out": "h5"},
                {"op": "emit", "answer": "h4", "evidence": "h5"},
            ],
        }
    )
    try:
        execution = execute_proofscript(proof, backend)
    except (KeyError, TypeError, ValueError):
        return None
    question = (
        f"Which page linked from {bridge.label} belongs to the Wikipedia category "
        f"'{category}'?"
    )
    challenge_id = f"kilt-selfplay-{bridge_id}-{target_id}-{_slug(category)}"
    example = BenchmarkExample(
        example_id=challenge_id,
        dataset="hotpotqa",
        split="selfplay",
        question=question,
        topic_entity_ids=(),
        gold_answers=execution.answers,
        answer_aliases=((target.label, *target.aliases),),
        gold_provenance=(execution.evidence,),
        metadata={
            "source_format": "evidence_selfplay_v1",
            "graph_snapshot": str(getattr(backend, "snapshot_id", "kilt-unknown")),
            "bridge_page_id": bridge_id,
            "target_page_id": target_id,
            "category": category,
        },
    )
    return EvidenceChallenge(
        challenge_id=challenge_id,
        bridge_page_id=bridge_id,
        target_page_id=target_id,
        category=category,
        example=example,
        proof=proof,
    )


def _policy_residual(
    challenge: EvidenceChallenge,
    proof: ProofExecution,
    run: EvidencePolicyRun,
) -> EvidenceResidual:
    observed_pages = {passage.page_id for passage in run.observed_passages}
    prefix = 0
    for evidence in proof.evidence:
        if evidence.page_id not in observed_pages:
            break
        prefix += 1
    missing = tuple(
        evidence.passage_key
        for evidence in proof.evidence
        if evidence.page_id not in observed_pages
    )
    predictions = {
        normalize_openqa_answer(str(answer.value))
        for answer in run.prediction.answer.answers
    }
    aliases = {
        normalize_openqa_answer(alias)
        for group in challenge.example.answer_aliases
        for alias in group
    }
    answer_f1 = 1.0 if predictions & aliases else 0.0
    kind: EvidenceResidualKind = (
        "retrieval" if missing else "reasoning" if answer_f1 < 1.0 else "success"
    )
    return EvidenceResidual(
        kind=kind,
        recovered_prefix=prefix,
        proof_length=len(proof.evidence),
        missing_passage_keys=missing,
        answer_f1=answer_f1,
    )


def _sft_row(record: EvidenceSelfPlayRecord, *, split: str) -> dict[str, object]:
    proof = record.proof_execution
    prompt = evidence_answer_prompt(
        record.challenge.example.question,
        record.evidence_flow_run.observed_passages,
        trace_id=f"selfplay:{record.challenge.challenge_id}:{split}",
    )
    answer = str(proof.answers.answers[0].value)
    response: dict[str, object] = {
        "answer": answer,
        "passage_keys": list(proof.passage_keys),
    }
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _compact_json(response)},
        ],
        "role": "solver",
        "task_id": record.challenge.challenge_id,
        "interaction_mode": "evidence_flow",
        "graphscript_version": "0.4",
        "operator_set": ["retrieve", "expand", "select_evidence", "answer"],
    }


def _write_sft_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with ParquetRowWriter(path, schema=SFT_SCHEMA, batch_size=64) as writer:
        for row in rows:
            writer.write(row)


def _category_leaks_answer(category_id: str, answer: str) -> bool:
    category_tokens = set(normalize_openqa_answer(category_id.removeprefix("category:")).split())
    answer_tokens = set(normalize_openqa_answer(answer).split())
    return bool((category_tokens - _WEAK_CATEGORY_TOKENS) & answer_tokens)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:48]


def _compact_json(value: dict[str, object]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
