from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from graphtask_r1.evaluation import KILTQAMetrics, kilt_qa_metrics
from graphtask_r1.experiments.evidence_selfplay import (
    EvidenceSelfPlayRecord,
    EvidenceSelfPlayRound,
)
from graphtask_r1.graph import GraphBackend
from graphtask_r1.graphscript import (
    BackendEvidenceRetriever,
    EvidenceEpisode,
    EvidenceFlowError,
    EvidenceTurn,
    ProofExecution,
    evidence_solver_prompt,
    execute_proofscript,
    execute_searchscript,
    parse_proofscript,
    parse_searchscript,
)
from graphtask_r1.rewards import (
    classify_evidence_residual,
    evidence_ecp_answer_gated_reward,
    evidence_ecp_causal_reward,
    evidence_ecp_curriculum_reward,
    evidence_ecp_protocol_gated_reward,
    evidence_ecp_solver_reward,
    evidence_questioner_reward,
    evidence_search_r1_em_reward,
    evidence_solver_reward,
)
from graphtask_r1.schema import AnswerSet, EvidenceProvenance, PassageHit, RewardBreakdown
from graphtask_r1.utils import (
    ParquetRowWriter,
    read_json,
    stable_hash,
    write_json,
    write_records,
)

SolverRewardVariant = Literal[
    "additive_v1",
    "search_r1_em",
    "ecp_v1",
    "ecp_v2",
    "ecp_v3",
    "ecp_v4",
    "ecp_v5",
]


class EvidenceQuestionerProposal(BaseModel):
    """Direct-RL proposal selecting a certified task from a private candidate pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["0.4-questioner"] = "0.4-questioner"
    challenge_id: str


def build_promoted_solver_curriculum(
    questioner_rows: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    completion_events: Sequence[Mapping[str, Any]],
    *,
    target_size: int,
    seed: int,
    retain_all_solver_rows: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Promote online Questioner proposals into a replayable Solver curriculum.

    A legal positive-reward rollout wins within its private candidate pool. If the
    Questioner never emitted a legal proposal for a pool, the certified frontier
    reward supplies a deterministic fallback. Promoted tasks are then replayed
    with frontier-weighted sampling, which closes the self-play loop without SFT.
    """

    pools: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = {}
    pool_settings: dict[tuple[str, ...], tuple[float, float]] = {}
    for row in questioner_rows:
        info = row["extra_info"]
        if not isinstance(info, Mapping):
            raise ValueError("questioner extra_info must be an object")
        raw_frontiers = info["candidate_frontiers"]
        if not isinstance(raw_frontiers, Mapping):
            raise ValueError("candidate_frontiers must be an object")
        frontiers = {
            str(challenge_id): value
            for challenge_id, value in raw_frontiers.items()
            if isinstance(value, Mapping)
        }
        signature = tuple(sorted(frontiers))
        pools[signature] = frontiers
        pool_settings[signature] = (
            float(info.get("frontier_target", 0.5)),
            float(info.get("frontier_sigma", 0.2)),
        )

    all_candidate_ids = {challenge_id for signature in pools for challenge_id in signature}
    accepted: dict[tuple[str, ...], tuple[float, str]] = {}
    rollout_count = 0
    accepted_count = 0
    for event in completion_events:
        prompts = event.get("prompt", [])
        completions = event.get("completion", [])
        rewards = event.get("GraphTaskReward", [])
        if not all(
            isinstance(values, Sequence) and not isinstance(values, str | bytes)
            for values in (prompts, completions, rewards)
        ):
            continue
        for prompt, completion, reward in zip(prompts, completions, rewards, strict=False):
            rollout_count += 1
            if not isinstance(prompt, str) or not isinstance(completion, str):
                continue
            signature = tuple(
                sorted(
                    challenge_id
                    for challenge_id in set(
                        re.findall(r'"challenge_id"\s*:\s*"([^"]+)"', prompt)
                    )
                    if challenge_id in all_candidate_ids
                )
            )
            if signature not in pools or float(reward) <= 0.0:
                continue
            try:
                proposal = EvidenceQuestionerProposal.model_validate_json(completion)
            except ValidationError:
                continue
            if proposal.challenge_id not in pools[signature]:
                continue
            accepted_count += 1
            candidate = (float(reward), proposal.challenge_id)
            if candidate > accepted.get(signature, (-math.inf, "")):
                accepted[signature] = candidate

    selections: list[dict[str, object]] = []
    for signature, frontiers in sorted(pools.items()):
        target, sigma = pool_settings[signature]
        scored = {
            challenge_id: evidence_questioner_reward(
                float(frontier["evidence_recovery_rate"]),
                float(frontier["conditional_answer_rate"]),
                novelty=float(frontier.get("novelty", 1.0)),
                target=target,
                sigma=sigma,
            ).total
            for challenge_id, frontier in frontiers.items()
        }
        if signature in accepted:
            rollout_reward, challenge_id = accepted[signature]
            source = "questioner_rollout"
        else:
            challenge_id = max(scored, key=lambda value: (scored[value], value))
            rollout_reward = None
            source = "certified_frontier_fallback"
        selections.append(
            {
                "pool": list(signature),
                "challenge_id": challenge_id,
                "selection_source": source,
                "rollout_reward": rollout_reward,
                "frontier_score": scored[challenge_id],
            }
        )

    solver_by_id: dict[str, Mapping[str, Any]] = {}
    for row in solver_rows:
        info = row["extra_info"]
        if isinstance(info, Mapping):
            solver_by_id[str(info["task_id"])] = row
    selected_ids = [str(item["challenge_id"]) for item in selections]
    rng = random.Random(seed)
    replay_ids = list(solver_by_id) if retain_all_solver_rows else list(selected_ids)
    if target_size < len(replay_ids):
        replay_ids = rng.choices(
            replay_ids,
            k=target_size,
        )
    elif target_size > len(replay_ids):
        replay_ids.extend(
            rng.choices(
                selected_ids,
                weights=[float(str(item["frontier_score"])) for item in selections],
                k=target_size - len(replay_ids),
            )
        )
    rng.shuffle(replay_ids)

    selection_by_id = {str(item["challenge_id"]): item for item in selections}
    occurrences: Counter[str] = Counter()
    curriculum: list[dict[str, object]] = []
    for challenge_id in replay_ids:
        occurrences[challenge_id] += 1
        curriculum_row: dict[str, object] = deepcopy(dict(solver_by_id[challenge_id]))
        curriculum_info = curriculum_row["extra_info"]
        assert isinstance(curriculum_info, dict)
        selection = selection_by_id.get(challenge_id)
        selection_source = (
            selection["selection_source"] if selection is not None else "coverage_replay"
        )
        frontier_score = selection["frontier_score"] if selection is not None else None
        curriculum_info.update(
            {
                "curriculum_round": 2,
                "archive_selection_source": selection_source,
                "archive_frontier_score": frontier_score,
                "archive_replay_index": occurrences[challenge_id],
            }
        )
        curriculum_row["uid"] = (
            f"evidence-solver-r2:{challenge_id}:{occurrences[challenge_id]}"
        )
        curriculum.append(curriculum_row)

    manifest: dict[str, object] = {
        "schema_version": "evidence-frontier-archive-v1",
        "algorithm": "online_questioner_proposal_archive_promotion",
        "uses_sft": False,
        "seed": seed,
        "rollouts_observed": rollout_count,
        "legal_positive_rollouts": accepted_count,
        "candidate_pools": len(pools),
        "rollout_promotions": sum(
            item["selection_source"] == "questioner_rollout" for item in selections
        ),
        "fallback_promotions": sum(
            item["selection_source"] == "certified_frontier_fallback"
            for item in selections
        ),
        "retains_all_solver_rows": retain_all_solver_rows,
        "coverage_replays": sum(
            item["extra_info"]["archive_selection_source"] == "coverage_replay"
            for item in curriculum
            if isinstance(item["extra_info"], Mapping)
        ),
        "solver_rows": len(curriculum),
        "selections": selections,
    }
    return curriculum, manifest


def evidence_questioner_prompt(
    candidates: Sequence[EvidenceSelfPlayRecord],
) -> list[dict[str, str]]:
    visible = [
        {
            "challenge_id": record.challenge.challenge_id,
            "question": record.challenge.example.question,
            "bridge_page_id": record.challenge.bridge_page_id,
            "target_category": record.challenge.category,
        }
        for record in candidates
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are the Questioner in adversarial self-play. Select one certified "
                "multi-hop challenge that is neither trivial nor currently impossible for "
                "the Solver. Emit only JSON: "
                '{"version":"0.4-questioner","challenge_id":"..."}. '
                "Solver scores and hidden ProofScripts are not visible to you."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"candidate_challenges": visible},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def export_evidence_rl_round(
    round_result: EvidenceSelfPlayRound,
    output_dir: Path,
    *,
    seed: int,
    train_ratio: float = 0.75,
    split_manifest: Path | None = None,
    questioner_weight: float = 1.0,
    solver_weight: float = 1.0,
    frontier_target: float = 0.5,
    frontier_sigma: float = 0.2,
    solver_reward_variant: SolverRewardVariant = "additive_v1",
) -> dict[str, object]:
    """Export direct-RL rows; no demonstration or SFT target is produced."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    records = list(round_result.records)
    train, validation, manifest = _split_records(
        records,
        seed=seed,
        train_ratio=train_ratio,
        split_manifest=split_manifest,
    )
    frequencies = Counter(record.challenge.challenge_id for record in train)
    grouped: dict[str, list[EvidenceSelfPlayRecord]] = defaultdict(list)
    for record in train:
        grouped[record.challenge.bridge_page_id].append(record)

    questioner_rows = [
        _questioner_row(
            candidates,
            frequencies=frequencies,
            questioner_weight=questioner_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
        )
        for _, candidates in sorted(grouped.items())
    ]
    solver_rows = [
        _solver_row(
            record,
            solver_weight=solver_weight,
            solver_reward_variant=solver_reward_variant,
        )
        for record in train
    ]
    validation_rows = [
        _solver_row(
            record,
            solver_weight=solver_weight,
            solver_reward_variant=solver_reward_variant,
        )
        for record in validation
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(
        output_dir / "records.parquet",
        (record.model_dump(mode="json") for record in records),
    )
    _write_rl_rows(output_dir / "questioner_train.parquet", questioner_rows)
    _write_rl_rows(output_dir / "solver_train.parquet", solver_rows)
    _write_rl_rows(output_dir / "solver_val.parquet", validation_rows)
    write_json(output_dir / "split.json", manifest)
    evaluation = {
        "train": _evaluation_split_summary(train),
        "validation": _evaluation_split_summary(validation),
    }
    write_json(output_dir / "evaluation.json", evaluation)
    summary: dict[str, object] = {
        "schema_version": "evidence-selfplay-rl-v1",
        "seed": seed,
        "questioner_rows": len(questioner_rows),
        "solver_train_rows": len(solver_rows),
        "solver_validation_rows": len(validation_rows),
        "uses_sft": False,
        "questioner_algorithm": "reinforce_plus_plus",
        "solver_algorithm": "grpo",
        "solver_reward_variant": solver_reward_variant,
        "ms_swift_version": "3.10.3",
        "evaluation": evaluation,
    }
    write_json(output_dir / "rl_manifest.json", summary)
    return summary


def compute_evidence_solver_score(
    solution: str,
    info: Mapping[str, Any],
    *,
    backend: GraphBackend,
) -> dict[str, float]:
    role_weight = float(info.get("role_weight", 1.0))
    try:
        proof = execute_proofscript(
            parse_proofscript(_required_string(info, "proof_json")), backend
        )
        rollout = info.get("solver_rollout")
        if isinstance(rollout, Mapping):
            return _compute_evidence_tool_score(
                solution,
                rollout,
                proof=proof,
                aliases=_answer_aliases(info),
                question=_required_string(info, "question"),
                role_weight=role_weight,
                reward_variant=_solver_reward_variant(info),
            )
        script = parse_searchscript(solution)
        episode = execute_searchscript(
            script,
            _required_string(info, "question"),
            BackendEvidenceRetriever(backend),
            trace_id=f"evidence-solver:{info.get('task_id', 'unknown')}",
        )
        aliases = _answer_aliases(info)
        reward = _evidence_reward(
            proof,
            episode,
            aliases,
            variant=_solver_reward_variant(info),
        )
        residual = classify_evidence_residual(proof, episode, aliases)
        return {
            "score": reward.total * role_weight,
            "raw_score": reward.total,
            **{name: float(value) for name, value in reward.components.items()},
            "proof_prefix_fraction": residual.prefix_fraction,
            "residual_retrieval": float(residual.kind == "retrieval"),
            "residual_reasoning": float(residual.kind == "reasoning"),
            "residual_success": float(residual.kind == "success"),
            "search_turns": float(len(episode.turns)),
        }
    except EvidenceFlowError as exc:
        return _rejection(role_weight, exc.reason_code)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
        return _rejection(role_weight, "INVALID_OUTPUT")


def compute_evidence_questioner_score(
    solution: str,
    info: Mapping[str, Any],
    *,
    backend: GraphBackend,
) -> dict[str, float]:
    role_weight = float(info.get("role_weight", 1.0))
    try:
        proposal = EvidenceQuestionerProposal.model_validate_json(solution)
        raw_candidates = info.get("candidate_frontiers")
        if not isinstance(raw_candidates, Mapping):
            raise ValueError("candidate_frontiers must be an object")
        raw_candidate = raw_candidates.get(proposal.challenge_id)
        if not isinstance(raw_candidate, Mapping):
            return _rejection(role_weight, "CANDIDATE_NOT_ALLOWED")
        proof = execute_proofscript(
            parse_proofscript(_required_string(raw_candidate, "proof_json")), backend
        )
        if not proof.answers.answers or not proof.evidence:
            return _rejection(role_weight, "UNCERTIFIED_PROOF")
        recovery = float(raw_candidate["evidence_recovery_rate"])
        conditional = float(raw_candidate["conditional_answer_rate"])
        novelty = float(raw_candidate.get("novelty", 1.0))
        reward = evidence_questioner_reward(
            recovery,
            conditional,
            novelty=novelty,
            target=float(info.get("frontier_target", 0.5)),
            sigma=float(info.get("frontier_sigma", 0.2)),
        )
        return {
            "score": reward.total * role_weight,
            "raw_score": reward.total,
            **{name: float(value) for name, value in reward.components.items()},
            "evidence_recovery_rate": recovery,
            "conditional_answer_rate": conditional,
            "proof_evidence_count": float(len(proof.evidence)),
        }
    except EvidenceFlowError as exc:
        return _rejection(role_weight, exc.reason_code)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
        return _rejection(role_weight, "INVALID_OUTPUT")


def _solver_row(
    record: EvidenceSelfPlayRecord,
    *,
    solver_weight: float,
    solver_reward_variant: SolverRewardVariant = "additive_v1",
) -> dict[str, object]:
    example = record.challenge.example
    aliases = example.answer_aliases or tuple(
        (str(answer.value),) for answer in example.gold_answers.answers
    )
    return {
        "data_source": "graphtask/evidence_solver",
        "prompt": evidence_solver_prompt(example.question),
        "ability": "certified_evidence_search",
        "reward_model": {"style": "rule", "ground_truth": "{}"},
        "extra_info": {
            "role": "evidence_solver",
            "role_weight": solver_weight,
            "task_id": record.challenge.challenge_id,
            "graph_snapshot": _training_snapshot(example.metadata),
            "interaction_mode": "tool",
            "graphscript_version": "0.4",
            "text_search_enabled": True,
            "evidence_reward_variant": solver_reward_variant,
            "question": example.question,
            "proof_json": record.challenge.proof.model_dump_json(by_alias=True),
            "answer_aliases": [list(group) for group in aliases],
        },
        "uid": f"evidence-solver:{record.challenge.challenge_id}",
    }


def _questioner_row(
    candidates: list[EvidenceSelfPlayRecord],
    *,
    frequencies: Counter[str],
    questioner_weight: float,
    frontier_target: float,
    frontier_sigma: float,
) -> dict[str, object]:
    candidates = sorted(candidates, key=lambda item: item.challenge.challenge_id)
    frontiers: dict[str, object] = {}
    for record in candidates:
        recovery = record.residual.prefix_fraction
        frontiers[record.challenge.challenge_id] = {
            "proof_json": record.challenge.proof.model_dump_json(by_alias=True),
            "evidence_recovery_rate": recovery,
            "conditional_answer_rate": (record.residual.answer_f1 if recovery == 1.0 else 0.0),
            "novelty": 1.0 / math.sqrt(frequencies[record.challenge.challenge_id]),
        }
    bridge_id = candidates[0].challenge.bridge_page_id
    snapshot = _training_snapshot(candidates[0].challenge.example.metadata)
    return {
        "data_source": "graphtask/evidence_questioner",
        "prompt": evidence_questioner_prompt(candidates),
        "ability": "certified_frontier_selection",
        "reward_model": {"style": "rule", "ground_truth": "{}"},
        "extra_info": {
            "role": "evidence_questioner",
            "role_weight": questioner_weight,
            "task_id": f"bridge:{bridge_id}",
            "graph_snapshot": snapshot,
            "interaction_mode": "graphscript",
            "graphscript_version": "0.4-questioner",
            "candidate_frontiers": frontiers,
            "frontier_target": frontier_target,
            "frontier_sigma": frontier_sigma,
        },
        "uid": f"evidence-questioner:{bridge_id}",
    }


def _split_records(
    records: list[EvidenceSelfPlayRecord],
    *,
    seed: int,
    train_ratio: float,
    split_manifest: Path | None,
) -> tuple[list[EvidenceSelfPlayRecord], list[EvidenceSelfPlayRecord], dict[str, object]]:
    by_id = {record.challenge.challenge_id: record for record in records}
    if split_manifest is not None and split_manifest.exists():
        raw = read_json(split_manifest)
        train = [by_id[str(value)] for value in raw["train_ids"] if str(value) in by_id]
        validation = [by_id[str(value)] for value in raw["validation_ids"] if str(value) in by_id]
    else:
        ranked = sorted(
            records,
            key=lambda record: (
                stable_hash([str(seed), record.challenge.challenge_id]),
                record.challenge.challenge_id,
            ),
        )
        train_count = max(1, min(len(ranked) - 1, round(len(ranked) * train_ratio)))
        train, validation = ranked[:train_count], ranked[train_count:]
    manifest: dict[str, object] = {
        "seed": seed,
        "train_ids": [record.challenge.challenge_id for record in train],
        "validation_ids": [record.challenge.challenge_id for record in validation],
    }
    return train, validation, manifest


def _write_rl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot export an empty direct-RL split: {path.name}")
    with ParquetRowWriter(path, batch_size=128) as writer:
        for row in rows:
            writer.write(row)


def _evaluation_split_summary(
    records: Sequence[EvidenceSelfPlayRecord],
) -> dict[str, object]:
    baseline = _mean_policy_metrics(records, policy="baseline")
    evidence_flow = _mean_policy_metrics(records, policy="evidence_flow")
    residual_counts = Counter(record.residual.kind for record in records)
    return {
        "examples": len(records),
        "recall_at": 5,
        "baseline": baseline,
        "evidence_flow": evidence_flow,
        "delta": {
            name: evidence_flow[name] - baseline[name]
            for name in KILTQAMetrics.model_fields
        },
        "residual_counts": {
            "retrieval": residual_counts["retrieval"],
            "reasoning": residual_counts["reasoning"],
            "success": residual_counts["success"],
        },
    }


def _mean_policy_metrics(
    records: Sequence[EvidenceSelfPlayRecord],
    *,
    policy: Literal["baseline", "evidence_flow"],
) -> dict[str, float]:
    metrics: list[KILTQAMetrics] = []
    for record in records:
        example = record.challenge.example
        aliases = example.answer_aliases or tuple(
            (str(answer.value),) for answer in example.gold_answers.answers
        )
        run = record.baseline_run if policy == "baseline" else record.evidence_flow_run
        metrics.append(
            kilt_qa_metrics(
                run.prediction.answer,
                run.provenance,
                aliases,
                example.gold_provenance,
                recall_at=5,
            )
        )
    return {
        name: sum(float(getattr(metric, name)) for metric in metrics) / len(metrics)
        for name in KILTQAMetrics.model_fields
    }


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _compute_evidence_tool_score(
    solution: str,
    rollout: Mapping[str, Any],
    *,
    proof: Any,
    aliases: tuple[tuple[str, ...], ...],
    question: str,
    role_weight: float,
    reward_variant: SolverRewardVariant,
) -> dict[str, float]:
    observed = tuple(
        PassageHit.model_validate(value) for value in rollout.get("observed_passages", [])
    )
    observed_by_key = {f"{passage.page_id}:{passage.paragraph_id}": passage for passage in observed}
    selected_keys = [str(value) for value in rollout.get("selected_passage_keys", [])]
    answer: str | None = None
    try:
        payload = json.loads(solution)
        if isinstance(payload, Mapping) and isinstance(payload.get("answer"), str):
            answer = str(payload["answer"]).strip()
            raw_keys = payload.get("passage_keys")
            if isinstance(raw_keys, Sequence) and not isinstance(raw_keys, str | bytes):
                selected_keys = [str(value) for value in raw_keys]
    except json.JSONDecodeError:
        pass
    selected = tuple(
        EvidenceProvenance(
            page_id=observed_by_key[key].page_id,
            paragraph_id=observed_by_key[key].paragraph_id,
            title=observed_by_key[key].title,
        )
        for key in dict.fromkeys(selected_keys)
        if key in observed_by_key
    )
    actions = rollout.get("evidence_actions", [])
    turns: list[EvidenceTurn] = []
    if isinstance(actions, Sequence) and not isinstance(actions, str | bytes):
        for index, raw in enumerate(actions):
            if not isinstance(raw, Mapping):
                continue
            passages = tuple(PassageHit.model_validate(value) for value in raw.get("passages", []))
            turns.append(EvidenceTurn(index=index, action=dict(raw), passages=passages))
    episode = EvidenceEpisode(
        question=question,
        turns=tuple(turns),
        final_answer=AnswerSet.literals([answer]) if answer else AnswerSet(),
        final_evidence=selected,
        done=bool(answer and selected),
    )
    if episode.done or reward_variant == "search_r1_em":
        reward = _evidence_reward(
            proof,
            episode,
            aliases,
            variant=reward_variant,
        )
        residual = classify_evidence_residual(proof, episode, aliases)
        return {
            "score": reward.total * role_weight,
            "raw_score": reward.total,
            **{name: float(value) for name, value in reward.components.items()},
            "proof_prefix_fraction": residual.prefix_fraction,
            "residual_retrieval": float(residual.kind == "retrieval"),
            "residual_reasoning": float(residual.kind == "reasoning"),
            "residual_success": float(residual.kind == "success"),
            "tool_calls": float(rollout.get("calls", 0)),
        }
    if reward_variant in {"ecp_v1", "ecp_v2", "ecp_v3", "ecp_v4", "ecp_v5"}:
        reward = _evidence_reward(proof, episode, aliases, variant=reward_variant)
        early_answers = int(rollout.get("early_answer_attempts", 0))
        invalid_calls = int(rollout.get("invalid_calls", 0))
        penalty = 0.025 * min(early_answers, 2) + 0.025 * min(invalid_calls, 2)
        raw_score = max(-0.2, reward.total - penalty)
        residual = classify_evidence_residual(proof, episode, aliases)
        return {
            "score": raw_score * role_weight,
            "raw_score": raw_score,
            **{name: float(value) for name, value in reward.components.items()},
            "proof_prefix_fraction": residual.prefix_fraction,
            "early_answer_penalty": float(0.025 * min(early_answers, 2)),
            "invalid_tool_penalty": float(0.025 * min(invalid_calls, 2)),
            "residual_incomplete": 1.0,
            "tool_calls": float(rollout.get("calls", 0)),
        }
    retrieved_keys = set(observed_by_key)
    prefix = 0
    for key in proof.passage_keys:
        if key not in retrieved_keys:
            break
        prefix += 1
    prefix_fraction = prefix / len(proof.passage_keys)
    calls = int(rollout.get("calls", 0))
    valid_calls = int(rollout.get("valid_calls", 0))
    valid_fraction = valid_calls / calls if calls else 0.0
    proof_keys = set(proof.passage_keys)
    selected_precision = (
        len({item.passage_key for item in selected} & proof_keys) / len(selected)
        if selected
        else 0.0
    )
    operation_types = {
        str(value.get("op"))
        for value in actions
        if isinstance(value, Mapping)
        and str(value.get("op")) in {"retrieve", "expand", "select_evidence"}
    }
    process_depth = min(calls, 4) / 4
    operation_coverage = len(operation_types) / 3
    proof_completion = float(prefix_fraction == 1.0 and "expand" in operation_types)
    early_answers = int(rollout.get("early_answer_attempts", 0))
    raw_score = (
        -0.2
        if calls == 0
        else -0.15
        + 0.05 * valid_fraction
        + 0.20 * prefix_fraction
        + 0.20 * selected_precision
        + 0.05 * float(answer is not None)
        + 0.10 * process_depth
        + 0.10 * operation_coverage
        + 0.30 * proof_completion
        - 0.05 * min(early_answers, 2)
    )
    return {
        "score": raw_score * role_weight,
        "raw_score": raw_score,
        "tool_call_attempted": float(calls > 0),
        "valid_tool_call_fraction": valid_fraction,
        "proof_prefix_fraction": prefix_fraction,
        "selected_proof_precision": selected_precision,
        "answer_format": float(answer is not None),
        "process_depth": process_depth,
        "operation_coverage": operation_coverage,
        "proof_completion": proof_completion,
        "early_answer_attempts": float(early_answers),
        "residual_incomplete": 1.0,
    }


def _training_snapshot(metadata: Mapping[str, Any]) -> str:
    snapshot = str(metadata.get("graph_snapshot", "kilt-2019-08-01-v1"))
    # Local KILT subsets share one snapshot contract; GRAPHTASK_KILT_DB selects
    # the concrete database instance used by reward workers.
    return "kilt-2019-08-01-v1" if snapshot.startswith("kilt-") else snapshot


def _answer_aliases(info: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    raw = info.get("answer_aliases")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("answer_aliases must be a list")
    aliases: list[tuple[str, ...]] = []
    for group in raw:
        if not isinstance(group, Sequence) or isinstance(group, str | bytes):
            raise ValueError("answer alias group must be a list")
        aliases.append(tuple(str(value) for value in group))
    return tuple(aliases)


def _solver_reward_variant(
    info: Mapping[str, Any],
) -> SolverRewardVariant:
    value = str(info.get("evidence_reward_variant", "additive_v1"))
    if value not in {
        "additive_v1",
        "search_r1_em",
        "ecp_v1",
        "ecp_v2",
        "ecp_v3",
        "ecp_v4",
        "ecp_v5",
    }:
        raise ValueError(f"unsupported evidence reward variant: {value}")
    return cast(SolverRewardVariant, value)


def _evidence_reward(
    proof: ProofExecution,
    episode: EvidenceEpisode,
    aliases: tuple[tuple[str, ...], ...],
    *,
    variant: SolverRewardVariant,
) -> RewardBreakdown:
    if variant == "search_r1_em":
        return evidence_search_r1_em_reward(proof, episode, aliases)
    if variant == "ecp_v5":
        return evidence_ecp_curriculum_reward(proof, episode, aliases)
    if variant == "ecp_v4":
        return evidence_ecp_causal_reward(proof, episode, aliases)
    if variant == "ecp_v3":
        return evidence_ecp_protocol_gated_reward(proof, episode, aliases)
    if variant == "ecp_v2":
        return evidence_ecp_answer_gated_reward(proof, episode, aliases)
    if variant == "ecp_v1":
        return evidence_ecp_solver_reward(proof, episode, aliases)
    return evidence_solver_reward(proof, episode, aliases)


def _rejection(role_weight: float, reason: str) -> dict[str, float]:
    raw_score = -0.2
    return {
        "score": raw_score * role_weight,
        "raw_score": raw_score,
        f"reject_{reason.casefold()}": 1.0,
    }
