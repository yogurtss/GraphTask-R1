from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from graphtask_r1.graph import GraphBackend
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.utils import file_hash, stable_hash, write_json
from kqapro_ere.mutation import DEFAULT_SLOT_PRIORITY
from kqapro_ere.residual import (
    ExecutionBudget,
    ResidualAnalysis,
    ResidualError,
    analyze_execution_residual,
)

_RENDERED_QUESTION = re.compile(
    r"<\|im_start\|>user\nQuestion: (.*?)\nAllowed relation catalog:",
    re.DOTALL,
)
_SLOT_FIELD = {
    "relation": "relation",
    "direction": "direction",
    "qualifier": "qualifier",
    "attribute": "attribute",
    "comparator": "comparator",
    "literal": "value",
    "mode": "mode",
    "type_id": "type_id",
    "query": "query",
}


@dataclass(frozen=True)
class SolverExample:
    task_id: str
    question: str
    reference_program_json: str
    allowed_relations: frozenset[str]
    seed_entity: str | None


@dataclass(frozen=True)
class CompletionObservation:
    step: int
    ordinal: int
    question: str
    completion: str


def _question_from_messages(messages: list[dict[str, object]]) -> str:
    user_content = next(
        str(message["content"]) for message in messages if message.get("role") == "user"
    )
    prefix = "Question: "
    suffix = "\nAllowed relation catalog:"
    return user_content.split(suffix, 1)[0].removeprefix(prefix)


def _load_solver_examples(path: Path) -> dict[str, SolverExample]:
    examples: dict[str, SolverExample] = {}
    for row in pq.read_table(path).to_pylist():
        prompt = list(row["prompt"])
        extra = dict(row["extra_info"])
        question = _question_from_messages(prompt)
        topic_entities = tuple(str(value) for value in extra["topic_entity_ids"])
        example = SolverExample(
            task_id=str(extra["task_id"]),
            question=question,
            reference_program_json=str(extra["reference_program_json"]),
            allowed_relations=frozenset(str(value) for value in extra["allowed_relations"]),
            seed_entity=topic_entities[0] if topic_entities else None,
        )
        existing = examples.get(question)
        if existing is not None and existing.task_id != example.task_id:
            raise ValueError(f"ambiguous solver question: {question}")
        examples[question] = example
    return examples


def _latest_completions(path: Path) -> tuple[list[CompletionObservation], int]:
    latest: dict[str, CompletionObservation] = {}
    total = 0
    ordinal = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            steps = list(record["step"])
            prompts = list(record["prompt"])
            completions = list(record["completion"])
            if not (len(steps) == len(prompts) == len(completions)):
                raise ValueError("completion artifact columns have different lengths")
            for raw_step, rendered_prompt, completion in zip(
                steps, prompts, completions, strict=True
            ):
                match = _RENDERED_QUESTION.search(str(rendered_prompt))
                if match is None:
                    raise ValueError("rendered completion prompt has no KQAPro question")
                observation = CompletionObservation(
                    step=int(raw_step),
                    ordinal=ordinal,
                    question=match.group(1),
                    completion=str(completion),
                )
                current = latest.get(observation.question)
                if current is None or (observation.step, observation.ordinal) > (
                    current.step,
                    current.ordinal,
                ):
                    latest[observation.question] = observation
                total += 1
                ordinal += 1
    return sorted(latest.values(), key=lambda value: value.ordinal), total


def _residual_slot(
    analysis: ResidualAnalysis,
) -> tuple[str | None, object | None, str]:
    residual = analysis.earliest_divergence
    if residual is None or residual.predicted_operation is None:
        return None, None, "RESIDUAL_UNRESOLVED"
    predicted = residual.predicted_operation
    gold = residual.gold_operation
    predicted_op = predicted.get("op")
    gold_op = gold.get("op")
    if predicted_op != gold_op:
        if predicted_op in {"intersect", "union"} and gold_op in {"intersect", "union"}:
            return "set_operator", predicted_op, "RESIDUAL_LOCALIZED"
        return None, None, "OPERATOR_MISMATCH"

    for slot in DEFAULT_SLOT_PRIORITY:
        if slot == "set_operator":
            continue
        field = _SLOT_FIELD.get(slot)
        if field is not None and predicted.get(field) != gold.get(field):
            return slot, predicted.get(field), "RESIDUAL_LOCALIZED"
    return None, None, "RESIDUAL_UNRESOLVED"


def mine_residual_profile(
    tasks_path: Path,
    completions_path: Path,
    relation_catalog_path: Path,
    output_dir: Path,
    *,
    backend: GraphBackend,
    seed: int,
    max_edge_visits: int,
    max_returned_entities: int,
) -> dict[str, object]:
    """Mine the latest observed Solver residual per train task into a replayable profile."""

    examples = _load_solver_examples(tasks_path)
    observations, total_observations = _latest_completions(completions_path)
    catalog = load_relation_catalog(relation_catalog_path)
    catalog_relations = frozenset(relation.relation_id for relation in catalog)
    budget = ExecutionBudget(
        max_edge_visits=max_edge_visits,
        max_returned_entities=max_returned_entities,
    )
    reason_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    effective_slot_counts: Counter[str] = Counter()
    observed_values: dict[str, dict[str, object]] = defaultdict(dict)
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals_path = output_dir / "residuals.jsonl"
    with residuals_path.open("w", encoding="utf-8") as stream:
        for index, observation in enumerate(observations):
            example = examples.get(observation.question)
            if example is None:
                reason_counts["TASK_NOT_FOUND"] += 1
                stream.write(
                    json.dumps(
                        {
                            "accepted": False,
                            "reason_code": "TASK_NOT_FOUND",
                            "step": observation.step,
                            "question": observation.question,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue
            allowed_relations = example.allowed_relations or catalog_relations
            try:
                analysis = analyze_execution_residual(
                    observation.completion,
                    example.reference_program_json,
                    backend=backend,
                    budget=budget,
                    allowed_relations=allowed_relations,
                    seed_entity=example.seed_entity,
                    seed=seed,
                    trace_id=f"ere-profile:{example.task_id}:{observation.step}:{index}",
                )
            except ResidualError as exc:
                reason_counts[exc.reason_code] += 1
                record: dict[str, Any] = {
                    "accepted": False,
                    "task_id": example.task_id,
                    "step": observation.step,
                    "reason_code": exc.reason_code,
                    "stage": exc.stage,
                    "detail": exc.detail,
                }
            else:
                if analysis.terminal_exact_match == 1.0:
                    reason = "TERMINAL_CORRECT"
                    slot = None
                    value = None
                else:
                    slot, value, reason = _residual_slot(analysis)
                reason_counts[reason] += 1
                if slot is not None:
                    slot_counts[slot] += 1
                    if value is not None:
                        effective_slot_counts[slot] += 1
                        observed_values[slot].setdefault(stable_hash(value), value)
                record = {
                    "accepted": True,
                    "task_id": example.task_id,
                    "step": observation.step,
                    "reason_code": reason,
                    "residual_slot": slot,
                    "effective_signal": slot is not None and value is not None,
                    "analysis": analysis.model_dump(mode="json"),
                }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    default_order = {slot: index for index, slot in enumerate(DEFAULT_SLOT_PRIORITY)}
    slot_priority = sorted(
        DEFAULT_SLOT_PRIORITY,
        key=lambda slot: (-effective_slot_counts[slot], default_order[slot]),
    )
    localized_residual_count = sum(slot_counts.values())
    effective_signal_count = sum(effective_slot_counts.values())
    profile: dict[str, object] = {
        "algorithm": "execution_residual_evolution_v1",
        "selection": "latest_completion_per_question",
        "tasks": str(tasks_path),
        "tasks_sha256": file_hash(tasks_path),
        "completions": str(completions_path),
        "completions_sha256": file_hash(completions_path),
        "relation_catalog": str(relation_catalog_path),
        "relation_catalog_sha256": file_hash(relation_catalog_path),
        "seed": seed,
        "total_completion_observations": total_observations,
        "selected_latest_observations": len(observations),
        "slot_priority": slot_priority,
        "slot_counts": dict(sorted(slot_counts.items())),
        "localized_residual_count": localized_residual_count,
        "effective_slot_counts": dict(sorted(effective_slot_counts.items())),
        "effective_signal_count": effective_signal_count,
        "has_effective_signal": effective_signal_count > 0,
        "observed_values": {
            slot: [values[key] for key in sorted(values)]
            for slot, values in sorted(observed_values.items())
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "budget": budget.model_dump(mode="json"),
        "residuals": str(residuals_path),
        "residuals_sha256": file_hash(residuals_path),
    }
    write_json(output_dir / "residual_profile.json", profile)
    return profile
