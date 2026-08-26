from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from graphtask_r1.archive import TaskArchive
from graphtask_r1.generation import verbalize
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import program_to_graphscript
from graphtask_r1.schema import Program, TaskProposal, TaskTrainingRecord, parse_program
from graphtask_r1.training.opponent import request_opponent
from graphtask_r1.training.sft_dataset import SFT_SCHEMA
from graphtask_r1.utils import ParquetRowWriter, iter_record_json, stable_hash, write_json
from graphtask_r1.verification import verify_task

RULE_QUESTIONER_VARIANT = "rule_program_question_v1"
RULE_QUESTIONER_SYSTEM_PROMPT = """You are the linguistic Questioner in graph self-play. The
system has already sampled and certified the executable GraphScript. Write one natural-language
question whose denotation is exactly that program. Return exactly one JSON object in the shape
{"question":"..."} with no prose or markdown. Do not copy IDs mechanically when labels are
available, do not reveal or guess the executed answer, and do not alter or reproduce the program."""

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "are",
        "by",
        "does",
        "from",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "which",
        "who",
    }
)


def rule_questioner_messages(
    program: Program,
    *,
    topic_entities: tuple[str, ...],
    backend: GraphBackend,
    target_question: str | None = None,
) -> list[dict[str, str]]:
    script = program_to_graphscript(program, version="0.3")
    relations = sorted(_program_relations(program))
    payload = {
        "seed_entities": [
            {
                "entity_id": entity_id,
                "label": backend.entity_info(entity_id).label,
            }
            for entity_id in topic_entities
        ],
        "relation_labels": {
            relation: backend.relation_info(relation).label for relation in relations
        },
        "certified_graphscript": script.model_dump(mode="json", by_alias=True),
    }
    messages = [
        {"role": "system", "content": RULE_QUESTIONER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    if target_question is not None:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"question": target_question},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return messages


def export_rule_questioner_sft(
    tasks_path: Path,
    output_path: Path,
    *,
    backend: GraphBackend,
    count: int,
    seed: int,
    max_topic_entities: int = 4,
) -> dict[str, object]:
    if count < 1:
        raise ValueError("count must be positive")
    eligible: list[TaskTrainingRecord] = []
    scanned = 0
    for raw in iter_record_json(tasks_path):
        scanned += 1
        task = TaskTrainingRecord.model_validate_json(raw)
        if not 0 < len(task.topic_entities) <= max_topic_entities:
            continue
        try:
            program_to_graphscript(task.program, version="0.3")
        except ValueError:
            continue
        eligible.append(task)
    ranked = sorted(
        eligible,
        key=lambda task: (
            stable_hash([RULE_QUESTIONER_VARIANT, str(seed), task.task_id]),
            task.task_id,
        ),
    )
    selected = ranked[:count]
    with ParquetRowWriter(output_path, schema=SFT_SCHEMA, batch_size=256) as writer:
        for task in selected:
            topic_ids = tuple(entity.entity_id for entity in task.topic_entities)
            writer.write(
                {
                    "messages": rule_questioner_messages(
                        task.program,
                        topic_entities=topic_ids,
                        backend=backend,
                        target_question=task.question,
                    ),
                    "role": "questioner",
                    "task_id": task.task_id,
                    "interaction_mode": "graphscript",
                    "graphscript_version": "0.3",
                    "operator_set": ["question"],
                }
            )
    metrics: dict[str, object] = {
        "variant": RULE_QUESTIONER_VARIANT,
        "scanned": scanned,
        "eligible": len(eligible),
        "requested": count,
        "selected": len(selected),
        "shortfall": max(0, count - len(selected)),
        "seed": seed,
        "output": str(output_path),
    }
    write_json(output_path.with_suffix(".metrics.json"), metrics)
    return metrics


def export_rule_questioner_rl(
    candidates_path: Path,
    output_path: Path,
    *,
    backend: GraphBackend,
    graph_snapshot: str,
    opponent_url: str,
    opponent_samples: int,
    count: int | None = None,
    round_index: int = 1,
    seed: int = 42,
) -> dict[str, object]:
    if opponent_samples < 1:
        raise ValueError("opponent_samples must be positive")
    if count is not None and count < 1:
        raise ValueError("count must be positive when provided")
    rows: list[dict[str, object]] = []
    scanned = 0
    rejected_uncertified = 0
    with candidates_path.open(encoding="utf-8") as stream:
        for line in stream:
            if count is not None and len(rows) >= count:
                break
            stripped = line.strip()
            if not stripped:
                continue
            scanned += 1
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError("candidate JSONL entries must be objects")
            if not bool(raw.get("strict_certified")):
                rejected_uncertified += 1
                continue
            program = parse_program(raw.get("program"))
            topic_ids = tuple(str(value) for value in raw.get("topic_entities", []))
            graphscript = program_to_graphscript(program, version="0.3")
            relations = sorted(_program_relations(program))
            task_id = f"rule-question-{round_index}-{len(rows):08d}"
            rows.append(
                {
                    "data_source": "graphtask/questioner",
                    "prompt": rule_questioner_messages(
                        program,
                        topic_entities=topic_ids,
                        backend=backend,
                    ),
                    "ability": "graph_question_verbalization",
                    "reward_model": {"style": "rule", "ground_truth": "{}"},
                    "extra_info": {
                        "role": "questioner",
                        "role_weight": 1.0,
                        "questioner_reward_variant": RULE_QUESTIONER_VARIANT,
                        "graph_snapshot": graph_snapshot,
                        "task_id": task_id,
                        "round": round_index,
                        "topic_entity_ids": list(topic_ids),
                        "fixed_program_json": json.dumps(
                            program.model_dump(mode="json"),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "fixed_graphscript_json": graphscript.model_dump_json(by_alias=True),
                        "allowed_relations": relations,
                        "opponent_url": opponent_url,
                        "opponent_samples": opponent_samples,
                        "opponent_seed": seed,
                        "frontier_target": 0.5,
                        "frontier_sigma": 0.2,
                        "question_alignment_min": 0.4,
                        "max_follow_limit": 100,
                        "max_edge_visits": 200,
                        "interaction_mode": "graphscript",
                        "graphscript_version": "0.3",
                    },
                    "uid": f"questioner:{task_id}",
                }
            )
    if not rows:
        raise ValueError("no strictly certified rule candidates available for RL export")
    with ParquetRowWriter(output_path, batch_size=256) as writer:
        for row in rows:
            writer.write(row)
    metrics: dict[str, object] = {
        "variant": RULE_QUESTIONER_VARIANT,
        "scanned": scanned,
        "rejected_uncertified": rejected_uncertified,
        "selected": len(rows),
        "requested": count,
        "round": round_index,
        "seed": seed,
        "output": str(output_path),
    }
    write_json(output_path.with_suffix(".metrics.json"), metrics)
    return metrics


def build_rule_questioner_mixed_sft(
    baseline_mixed_path: Path,
    questioner_path: Path,
    output_path: Path,
    *,
    seed: int,
) -> dict[str, object]:
    """Replace baseline Questioner rows while preserving its exact Solver rows."""

    baseline = pq.read_table(baseline_mixed_path)
    questioner = pq.read_table(questioner_path)
    solver = baseline.filter(pc.equal(baseline["role"], "solver"))
    baseline_questioner = baseline.filter(pc.equal(baseline["role"], "questioner"))
    if len(solver) + len(baseline_questioner) != len(baseline):
        raise ValueError("baseline mixed SFT contains roles other than solver/questioner")
    if set(questioner["role"].to_pylist()) != {"questioner"}:
        raise ValueError("replacement Questioner SFT must contain only questioner rows")
    if len(questioner) != len(baseline_questioner):
        raise ValueError(
            "replacement Questioner count must match baseline for a controlled A/B: "
            f"expected {len(baseline_questioner)}, got {len(questioner)}"
        )
    combined = pa.concat_tables([solver, questioner], promote_options="default")
    order = list(range(len(combined)))
    random.Random(seed).shuffle(order)
    combined = combined.take(pa.array(order, type=pa.int64()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, output_path)
    metrics: dict[str, object] = {
        "variant": RULE_QUESTIONER_VARIANT,
        "baseline": str(baseline_mixed_path),
        "solver_rows": len(solver),
        "baseline_questioner_rows": len(baseline_questioner),
        "replacement_questioner_rows": len(questioner),
        "total": len(combined),
        "seed": seed,
        "output": str(output_path),
    }
    write_json(output_path.with_suffix(".metrics.json"), metrics)
    return metrics


def rule_questioner_replacement_count(baseline_mixed_path: Path) -> int:
    """Return the exact Questioner count required for a controlled mixed-SFT A/B."""

    baseline = pq.read_table(baseline_mixed_path, columns=["role"])
    roles = baseline["role"].to_pylist()
    unexpected = sorted({str(role) for role in roles if role not in {"solver", "questioner"}})
    if unexpected:
        raise ValueError(
            "baseline mixed SFT contains roles other than solver/questioner: "
            + ", ".join(unexpected)
        )
    count = sum(1 for role in roles if role == "questioner")
    if count < 1:
        raise ValueError("baseline mixed SFT contains no Questioner rows")
    return count


def promote_rule_questioner_candidates(
    staged_path: Path,
    archive_path: Path,
    *,
    min_difficulty: float,
    max_difficulty: float,
    min_novelty: float = 0.0,
) -> dict[str, object]:
    """Admit fixed-program tasks using a signal compatible with one opponent sample.

    Exact-match pass rate is binary when a deterministic local opponent is sampled once,
    so a strict interval such as [0.25, 0.75] can never admit a task.  The independent
    experiment instead averages the three nested Solver milestones: parse, execute, and
    semantic F1.  With one sample this gives useful levels 0, 1/3, 2/3, and 1.
    """

    if not 0.0 <= min_difficulty <= max_difficulty <= 1.0:
        raise ValueError("difficulty bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= min_novelty <= 1.0:
        raise ValueError("min_novelty must be between 0 and 1")
    with TaskArchive(staged_path) as staged:
        candidates = sorted(
            staged.all(),
            key=lambda task: (task.program_signature, task.task_id),
        )
    decisions: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    accepted = 0
    with TaskArchive(archive_path) as archive:
        for task in candidates:
            stats = task.solver_stats
            parse_rate = float(stats.get("program_parse_rate", 0.0))
            execution_rate = float(stats.get("program_execution_rate", 0.0))
            semantic_rate = float(stats.get("mean_f1", 0.0))
            difficulty = (parse_rate + execution_rate + semantic_rate) / 3.0
            structural, textual = archive.novelty(task.program_signature, task.question)
            novelty = 0.5 * (structural + textual)
            reasons: list[str] = []
            if difficulty < min_difficulty:
                reasons.append("TOO_HARD")
            if difficulty > max_difficulty:
                reasons.append("TOO_EASY")
            if structural == 0.0:
                reasons.append("DUPLICATE_SIGNATURE")
            if novelty < min_novelty:
                reasons.append("LOW_NOVELTY")
            if not reasons:
                admission = {
                    "accepted": True,
                    "difficulty_signal": difficulty,
                    "program_parse_rate": parse_rate,
                    "program_execution_rate": execution_rate,
                    "semantic_f1": semantic_rate,
                    "novelty_structural": structural,
                    "novelty_textual": textual,
                    "novelty": novelty,
                    "reason_codes": [],
                }
                promoted = task.model_copy(
                    update={
                        "solver_stats": {
                            **stats,
                            "archive_admission": admission,
                        }
                    }
                )
                if archive.add(promoted):
                    accepted += 1
                else:
                    reasons.append("DUPLICATE_SIGNATURE")
            reason_counts.update(reasons)
            decisions.append(
                {
                    "task_id": task.task_id,
                    "program_signature": task.program_signature,
                    "accepted": not reasons,
                    "difficulty_signal": difficulty,
                    "novelty": novelty,
                    "reason_codes": reasons,
                }
            )
    return {
        "variant": RULE_QUESTIONER_VARIANT,
        "candidates": len(candidates),
        "accepted": accepted,
        "rejected": len(candidates) - accepted,
        "reason_counts": dict(sorted(reason_counts.items())),
        "thresholds": {
            "min_difficulty": min_difficulty,
            "max_difficulty": max_difficulty,
            "min_novelty": min_novelty,
        },
        "decisions": decisions,
    }


async def compute_rule_questioner_score(
    solution_str: str,
    info: dict[str, Any],
) -> dict[str, float]:
    values: dict[str, float] = {
        "json_valid": 0.0,
        "question_present": 0.0,
        "question_program_alignment": 0.0,
        "no_answer_leak": 0.0,
        "certified": 0.0,
        "opponent_parse_rate": 0.0,
        "opponent_execution_rate_given_parse": 0.0,
        "opponent_semantic_success_given_execution": 0.0,
        "frontier_reward": 0.0,
        "novelty_textual": 0.0,
    }
    rejection_reasons: list[str] = []
    payload = _extract_question_payload(solution_str)
    if payload is None:
        return _rule_score(values, info, ("NON_JSON",))
    values["json_valid"] = 1.0
    if set(payload) != {"question"}:
        return _rule_score(values, info, ("INVALID_OUTPUT",))
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return _rule_score(values, info, ("MISSING_QUESTION",))
    question = question.strip()
    values["question_present"] = 1.0
    raw_program = info.get("fixed_program_json")
    if not isinstance(raw_program, str):
        raise ValueError("rule Questioner reward requires fixed_program_json")
    program = parse_program(json.loads(raw_program))
    backend = backend_from_snapshot(str(info.get("graph_snapshot", "toy-v1")))
    canonical = verbalize(program, backend)
    alignment, token_f1, anchor_overlap = _question_alignment(question, canonical)
    values["question_program_alignment"] = alignment
    values["question_alignment_token_f1"] = token_f1
    values["question_alignment_anchor_overlap"] = anchor_overlap
    verification = verify_task(question, program, backend)
    values["no_answer_leak"] = float(not verification.answer_leak)
    values["certified"] = float(verification.passed)
    rejection_reasons.extend(verification.rejection_reasons)
    eligible = (
        verification.passed
        and alignment >= float(info.get("question_alignment_min", 0.4))
    )
    opponent_url = str(info.get("opponent_url") or "")
    if eligible and opponent_url:
        topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
        proposal = TaskProposal(topic_entities=topic_ids, program=program, paraphrase=question)
        evaluation = await request_opponent(
            opponent_url,
            proposal=proposal,
            graph_snapshot=str(info.get("graph_snapshot", "toy-v1")),
            samples=int(info.get("opponent_samples", 4)),
            round_index=int(info.get("round", 1)),
            interaction_mode="graphscript",
            graphscript_version="0.3",
            allowed_relations=tuple(str(value) for value in info.get("allowed_relations", [])),
            max_follow_limit=int(info.get("max_follow_limit", 100)),
            max_edge_visits=int(info.get("max_edge_visits", 200)),
            seed=int(info.get("opponent_seed", 42)),
            generated_question=question,
            recover_invalid_tool_calls=True,
        )
        semantic_success = float(evaluation["semantic_success_given_execution"])
        values.update(
            {
                "opponent_parse_rate": float(evaluation["program_parse_rate"]),
                "opponent_execution_rate_given_parse": float(
                    evaluation["execution_rate_given_parse"]
                ),
                "opponent_semantic_success_given_execution": semantic_success,
                "frontier_reward": _frontier_reward(
                    semantic_success,
                    target=float(info.get("frontier_target", 0.5)),
                    sigma=float(info.get("frontier_sigma", 0.2)),
                ),
                "novelty_textual": float(evaluation["novelty_textual"]),
            }
        )
    elif alignment < float(info.get("question_alignment_min", 0.4)):
        rejection_reasons.append("QUESTION_PROGRAM_MISMATCH")
    return _rule_score(values, info, tuple(dict.fromkeys(rejection_reasons)))


def _extract_question_payload(text: str) -> dict[str, object] | None:
    """Extract the first complete question-only JSON object from serving wrappers."""

    decoder = json.JSONDecoder()
    search_from = 0
    fallback: dict[str, object] | None = None
    while True:
        object_start = text.find("{", search_from)
        if object_start < 0:
            return fallback
        try:
            payload, _ = decoder.raw_decode(text, object_start)
        except json.JSONDecodeError:
            search_from = object_start + 1
            continue
        if isinstance(payload, dict):
            fallback = fallback or payload
            if set(payload) == {"question"}:
                return payload
        search_from = object_start + 1


def _rule_score(
    values: dict[str, float],
    info: dict[str, Any],
    rejection_reasons: tuple[str, ...],
) -> dict[str, float]:
    readiness = (
        values["opponent_parse_rate"] * values["opponent_execution_rate_given_parse"]
    )
    semantic_readiness = readiness * values["opponent_semantic_success_given_execution"]
    difficulty_signal = (
        values["opponent_parse_rate"] + readiness + semantic_readiness
    ) / 3.0
    score = (
        0.05 * values["json_valid"]
        + 0.05 * values["question_present"]
        + 0.20 * values["question_program_alignment"]
        + 0.10 * values["no_answer_leak"]
        + 0.10 * values["certified"]
        + 0.10 * values["opponent_parse_rate"]
        + 0.10 * readiness
        + 0.25 * readiness * values["frontier_reward"]
        + 0.05 * values["novelty_textual"]
    )
    role_weight = float(info.get("role_weight", 1.0))
    return {
        "score": score * role_weight,
        "raw_score": score,
        "opponent_difficulty_signal": difficulty_signal,
        **values,
        **{f"reject_{reason.lower()}": 1.0 for reason in rejection_reasons},
    }


def _question_alignment(generated: str, canonical: str) -> tuple[float, float, float]:
    generated_tokens = set(re.findall(r"[\w]+", generated.casefold()))
    canonical_tokens = set(re.findall(r"[\w]+", canonical.casefold()))
    overlap = len(generated_tokens & canonical_tokens)
    precision = overlap / len(generated_tokens) if generated_tokens else 0.0
    recall = overlap / len(canonical_tokens) if canonical_tokens else 0.0
    token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    anchors = canonical_tokens - _STOPWORDS
    anchor_overlap = len(anchors & generated_tokens) / len(anchors) if anchors else 1.0
    return 0.5 * token_f1 + 0.5 * anchor_overlap, token_f1, anchor_overlap


def _frontier_reward(success: float, *, target: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("frontier_sigma must be positive")
    return math.exp(-((success - target) ** 2) / (2 * sigma**2))


def _program_relations(program: Program) -> set[str]:
    values: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key in ("relation", "attribute", "qualifier"):
                value = node.get(key)
                if isinstance(value, str):
                    values.add(value)
            for value in node.values():
                visit(value)
        elif isinstance(node, list | tuple):
            for value in node:
                visit(value)

    visit(program.model_dump(mode="json"))
    return values


def iter_rule_candidates(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("candidate JSONL entries must be objects")
            yield value
