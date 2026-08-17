from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, Literal, TypeVar

from graphtask_r1.dsl import operator_tags
from graphtask_r1.schema import AnswerSet, Program, TaskCertificate, TaskTrainingRecord
from graphtask_r1.utils import stable_hash

QuestionerTask = TaskCertificate | TaskTrainingRecord
QuestionerTaskT = TypeVar("QuestionerTaskT", bound=QuestionerTask)
QUESTIONER_SAMPLER_VERSION = "explicit-root-structure-stratified-v1"
QUESTIONER_RANDOM_SAMPLER_VERSION = "explicit-root-random-v1"


def _program_node_count(value: object) -> int:
    if isinstance(value, dict):
        return int("op" in value) + sum(_program_node_count(item) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_program_node_count(item) for item in value)
    return 0


def _bucket(value: int, boundaries: tuple[int, ...]) -> str:
    lower = 1
    for upper in boundaries:
        if value <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def questioner_structure_profile(
    program: Program,
    *,
    root_count: int,
    answers: AnswerSet,
) -> dict[str, str]:
    nodes = _program_node_count(program.model_dump(mode="json"))
    answer_kinds = ",".join(sorted({answer.kind for answer in answers.answers}))
    return {
        "roots": _bucket(root_count, (1, 2, 4, 8)),
        "terminal": program.op,
        "nodes": _bucket(nodes, (3, 6, 10)),
        "ops": ",".join(operator_tags(program)),
        "answers": answer_kinds or "empty",
    }


def _encode_profile(profile: dict[str, str]) -> str:
    keys = ("roots", "terminal", "nodes", "ops", "answers")
    return "|".join(f"{key}={profile[key]}" for key in keys)


def questioner_task_stratum(task: QuestionerTask) -> str:
    """Describe target-relevant structure without exposing the gold program to the prompt."""
    return _encode_profile(
        questioner_structure_profile(
            task.program,
            root_count=len(task.topic_entities),
            answers=task.gold_answers,
        )
    )


def _structure_marginals(tasks: Iterable[QuestionerTask]) -> dict[str, Counter[str]]:
    marginals: dict[str, Counter[str]] = {
        "root_count": Counter(),
        "root_bucket": Counter(),
        "terminal": Counter(),
        "node_bucket": Counter(),
        "answer_kinds": Counter(),
        "operator_presence": Counter(),
    }
    for task in tasks:
        profile = questioner_structure_profile(
            task.program,
            root_count=len(task.topic_entities),
            answers=task.gold_answers,
        )
        marginals["root_count"][str(len(task.topic_entities))] += 1
        marginals["root_bucket"][profile["roots"]] += 1
        marginals["terminal"][profile["terminal"]] += 1
        marginals["node_bucket"][profile["nodes"]] += 1
        marginals["answer_kinds"][profile["answers"]] += 1
        for operator in (value for value in profile["ops"].split(",") if value):
            marginals["operator_presence"][operator] += 1
    return marginals


def target_structure_alignment(
    target_stratum: str,
    *,
    program: Program,
    root_count: int,
    answers: AnswerSet,
) -> dict[str, float]:
    """Score generated structure against a hidden train-derived target stratum."""
    try:
        target = dict(part.split("=", 1) for part in target_stratum.split("|"))
    except ValueError:
        return {"target_alignment": 0.0}
    generated = questioner_structure_profile(
        program,
        root_count=root_count,
        answers=answers,
    )
    target_ops = {value for value in target.get("ops", "").split(",") if value}
    generated_ops = {value for value in generated["ops"].split(",") if value}
    union = target_ops | generated_ops
    components = {
        "target_root_match": float(target.get("roots") == generated["roots"]),
        "target_terminal_match": float(target.get("terminal") == generated["terminal"]),
        "target_length_match": float(target.get("nodes") == generated["nodes"]),
        "target_operator_jaccard": len(target_ops & generated_ops) / len(union) if union else 1.0,
        "target_answer_match": float(target.get("answers") == generated["answers"]),
    }
    components["target_alignment"] = sum(components.values()) / len(components)
    return components


def _proportional_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    population = sum(counts.values())
    target = min(target, population)
    if target < 1:
        return {key: 0 for key in counts}
    ideals = {key: target * count / population for key, count in counts.items()}
    quotas = {key: min(counts[key], int(ideal)) for key, ideal in ideals.items()}
    remaining = target - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda key: (-(ideals[key] - quotas[key]), key),
    )
    for key in ranked:
        if not remaining:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("failed to allocate all Questioner sampling quotas")
    return quotas


def select_questioner_tasks(
    tasks: Iterable[QuestionerTaskT],
    *,
    count: int,
    seed: int,
    allow_oversample: bool = False,
    max_topic_entities: int | None = None,
    selection_mode: Literal["stratified", "random"] = "stratified",
) -> tuple[list[QuestionerTaskT], dict[str, Any]]:
    """Select explicit-root tasks without replacement unless oversampling is explicit."""
    if count < 1:
        raise ValueError("Questioner task count must be positive")
    if max_topic_entities is not None and max_topic_entities < 1:
        raise ValueError("max_topic_entities must be positive when provided")
    eligible: list[tuple[int, QuestionerTaskT, str]] = []
    source_strata: Counter[str] = Counter()
    scanned = 0
    rootless = 0
    too_many_roots = 0
    for index, task in enumerate(tasks):
        scanned = index + 1
        if not task.topic_entities:
            rootless += 1
            continue
        if max_topic_entities is not None and len(task.topic_entities) > max_topic_entities:
            too_many_roots += 1
            continue
        stratum = questioner_task_stratum(task)
        eligible.append((index, task, stratum))
        source_strata[stratum] += 1
    if not eligible:
        raise ValueError("no explicit-root tasks are eligible for Questioner data")
    unique_target = min(count, len(eligible))
    selected_with_strata: list[tuple[int, QuestionerTaskT, str]] = []
    if selection_mode == "stratified":
        quotas = _proportional_quotas(source_strata, unique_target)
        grouped: dict[str, list[tuple[int, QuestionerTaskT]]] = defaultdict(list)
        for index, task, stratum in eligible:
            grouped[stratum].append((index, task))
        for stratum in sorted(grouped):
            ranked = sorted(
                grouped[stratum],
                key=lambda item: (
                    stable_hash(
                        [
                            QUESTIONER_SAMPLER_VERSION,
                            str(seed),
                            item[1].task_id,
                            str(item[0]),
                        ]
                    ),
                    item[0],
                ),
            )
            selected_with_strata.extend(
                (index, task, stratum) for index, task in ranked[: quotas[stratum]]
            )
    else:
        ranked_random = sorted(
            eligible,
            key=lambda item: (
                stable_hash(
                    [
                        QUESTIONER_RANDOM_SAMPLER_VERSION,
                        str(seed),
                        item[1].task_id,
                        str(item[0]),
                    ]
                ),
                item[0],
            ),
        )
        selected_with_strata = ranked_random[:unique_target]
    selected_with_strata.sort(key=lambda item: item[0])
    unique_selected_strata = Counter(stratum for _, _, stratum in selected_with_strata)
    selected = [task for _, task, _ in selected_with_strata]
    selected_strata = [stratum for _, _, stratum in selected_with_strata]
    rng = random.Random(seed)
    if allow_oversample and count > len(selected):
        repeats, remainder = divmod(count, len(selected))
        selected = selected * repeats
        selected_strata = selected_strata * repeats
        if remainder:
            indices = rng.sample(range(unique_target), remainder)
            selected.extend(selected[index] for index in indices)
            selected_strata.extend(selected_strata[index] for index in indices)
        paired = list(zip(selected, selected_strata, strict=True))
        rng.shuffle(paired)
        selected = [task for task, _ in paired]
        selected_strata = [stratum for _, stratum in paired]

    final_strata = Counter(selected_strata)
    source_total = sum(source_strata.values())
    selected_total = sum(final_strata.values())
    source_marginals = _structure_marginals(task for _, task, _ in eligible)
    selected_marginals = _structure_marginals(selected)
    total_variation = 0.5 * sum(
        abs(source_strata[key] / source_total - final_strata[key] / selected_total)
        for key in source_strata
    )
    metrics: dict[str, Any] = {
        "sampler": (
            QUESTIONER_SAMPLER_VERSION
            if selection_mode == "stratified"
            else QUESTIONER_RANDOM_SAMPLER_VERSION
        ),
        "selection_mode": selection_mode,
        "seed": seed,
        "requested": count,
        "scanned": scanned,
        "rootless_rejected": rootless,
        "too_many_roots_rejected": too_many_roots,
        "eligible": len(eligible),
        "unique_selected": unique_target,
        "repeated_rows": max(0, len(selected) - unique_target),
        "shortfall": max(0, count - len(selected)),
        "selected": len(selected),
        "source_strata": len(source_strata),
        "selected_strata": len(final_strata),
        "distribution_total_variation": total_variation,
        "marginals": {
            dimension: {
                value: {
                    "source": source_marginals[dimension][value],
                    "source_prevalence": source_marginals[dimension][value] / source_total,
                    "final_selected": selected_marginals[dimension][value],
                    "final_prevalence": selected_marginals[dimension][value] / selected_total,
                }
                for value in sorted(
                    source_marginals[dimension] | selected_marginals[dimension]
                )
            }
            for dimension in source_marginals
        },
        "strata": {
            key: {
                "source": source_strata[key],
                "source_fraction": source_strata[key] / source_total,
                    "unique_selected": unique_selected_strata[key],
                "final_selected": final_strata[key],
                "final_fraction": final_strata[key] / selected_total,
            }
            for key in sorted(source_strata)
        },
    }
    return selected, metrics
