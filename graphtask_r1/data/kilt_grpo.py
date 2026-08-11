from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from graphtask_r1.dsl import canonical_signature
from graphtask_r1.generation import TraceCompilationError, certify_proposal, compile_trace
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.schema import (
    Count,
    Entity,
    FilterType,
    Hop,
    Program,
    TaskCertificate,
    TaskProposal,
    TaskProvenance,
    program_to_dict,
)
from graphtask_r1.training.prompts import GraphScriptVersion, InteractionMode
from graphtask_r1.training.relations import program_relations
from graphtask_r1.training.rl_dataset import export_role_dataset
from graphtask_r1.utils import (
    ProgressLogger,
    stable_hash,
    write_json,
    write_manifest,
    write_records,
)

KILT_BOOTSTRAP_VERSION = "kilt-certified-grpo-v2"
KiltProgramFamily = Literal["hop1", "hop2", "type_filter", "count"]
DEFAULT_KILT_FAMILIES: tuple[KiltProgramFamily, ...] = (
    "hop1",
    "hop2",
    "type_filter",
    "count",
)


@runtime_checkable
class _BulkDegreeBackend(Protocol):
    def entity_degrees(self, entity_ids: list[str]) -> dict[str, int]: ...


def _reason_codes(exc: Exception) -> tuple[str, ...]:
    if isinstance(exc, TraceCompilationError):
        return (exc.reason_code,)
    detail = str(exc)
    prefix = "proposal rejected: "
    if detail.startswith(prefix):
        values = tuple(value for value in detail.removeprefix(prefix).split(",") if value)
        if values:
            return values
    return (type(exc).__name__.upper(),)


def _candidate_program(
    backend: GraphBackend,
    root_id: str,
    family: KiltProgramFamily,
    rng: random.Random,
) -> tuple[Program | None, str | None]:
    first_direction: Literal["out", "in"] = rng.choice(("out", "in"))
    first = Hop(
        input=Entity(entity_id=root_id),
        relation="wikipedia_link",
        direction=first_direction,
    )
    if family == "hop1":
        return first, None
    if family == "hop2":
        return (
            Hop(
                input=first,
                relation="wikipedia_link",
                direction=rng.choice(("out", "in")),
            ),
            None,
        )
    first_values = backend.execute_program(first).entity_ids()
    if not first_values:
        return None, "EMPTY_FIRST_HOP"
    if family == "count":
        return Count(input=first), None
    typed_values = [
        (entity_id, type_id)
        for entity_id in first_values
        for type_id in backend.entity_info(entity_id).type_ids
    ]
    if not typed_values:
        return None, "NO_OUTPUT_TYPE"
    _, selected_type = rng.choice(sorted(typed_values))
    return FilterType(input=first, type_id=selected_type), None


def _eligible_roots(
    backend: GraphBackend,
    *,
    pool_limit: int,
    min_degree: int,
    max_degree: int,
    rng: random.Random,
) -> list[str]:
    roots = list(backend.all_entities(limit=pool_limit))
    rng.shuffle(roots)
    selected: list[str] = []
    progress = ProgressLogger("data.bootstrap_kilt_grpo.filter_roots", total=len(roots))
    progress.start(min_degree=min_degree, max_degree=max_degree)
    degrees = (
        backend.entity_degrees(roots)
        if isinstance(backend, _BulkDegreeBackend)
        else {
            entity_id: len(backend.neighbors([entity_id], direction="both", limit=max_degree + 1))
            for entity_id in roots
        }
    )
    for index, entity_id in enumerate(roots, start=1):
        degree = degrees[entity_id]
        if min_degree <= degree <= max_degree:
            selected.append(entity_id)
        progress.update(index, eligible=len(selected))
    progress.finish(len(roots), eligible=len(selected))
    return selected


def _certified_task(
    backend: GraphBackend,
    program: Program,
    *,
    root_id: str,
    family: KiltProgramFamily,
    attempt: int,
    seed: int,
    snapshot: str,
) -> tuple[TaskCertificate, dict[str, Any]]:
    task = certify_proposal(
        TaskProposal(topic_entities=(root_id,), program=program),
        backend,
        graph_snapshot=snapshot,
        round_index=0,
    )
    signature = canonical_signature(program)
    task_id = "gt_kilt_" + stable_hash([snapshot, signature, task.question])[:20]
    task = task.model_copy(
        update={
            "task_id": task_id,
            "source": "kilt_bootstrap",
            "source_id": f"kilt:{root_id}:{attempt}",
            "provenance": TaskProvenance(
                dataset="kilt",
                raw_file="kilt_knowledgesource.json",
                raw_index=attempt,
                converter_version=KILT_BOOTSTRAP_VERSION,
            ),
            "generation": {
                "seed": seed,
                "attempt": attempt,
                "family": family,
                "root_entity_id": root_id,
                "graph_snapshot": snapshot,
                "bootstrap_version": KILT_BOOTSTRAP_VERSION,
            },
        }
    )
    trace = compile_trace(task_id, task.question, program, backend, seed=seed + attempt)
    if trace.final_answers != task.gold_answers:
        raise TraceCompilationError(
            "TRACE_REPLAY_MISMATCH", "canonical trace answers differ from certified answers"
        )
    return task, trace.model_dump(mode="json")


def _split_tasks(
    tasks: list[TaskCertificate],
    traces: dict[str, dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> dict[str, tuple[list[TaskCertificate], list[dict[str, Any]]]]:
    shuffled = list(tasks)
    random.Random(seed + 1).shuffle(shuffled)
    val_count = 0
    if len(shuffled) > 1 and val_ratio > 0:
        val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_ratio)))
    val_ids = {task.task_id for task in shuffled[:val_count]}
    result: dict[str, tuple[list[TaskCertificate], list[dict[str, Any]]]] = {}
    for split in ("train", "val"):
        selected = [
            task.model_copy(update={"split": split})
            for task in tasks
            if (task.task_id in val_ids) == (split == "val")
        ]
        selected.sort(key=lambda task: task.task_id)
        result[split] = (selected, [traces[task.task_id] for task in selected])
    return result


def bootstrap_kilt_grpo(
    output_dir: Path,
    *,
    snapshot: str = "kilt-2019-08-01-v1",
    count: int = 1_024,
    seed: int = 42,
    pool_limit: int = 100_000,
    max_attempts: int | None = None,
    min_degree: int = 2,
    max_degree: int = 100,
    val_ratio: float = 0.1,
    families: tuple[KiltProgramFamily, ...] = DEFAULT_KILT_FAMILIES,
    interaction_mode: InteractionMode = "graphscript",
    graphscript_version: GraphScriptVersion = "0.2",
) -> dict[str, Any]:
    """Generate replayable KILT tasks and export them through the existing Solver GRPO contract."""

    if count < 1:
        raise ValueError("count must be at least 1")
    if pool_limit < 1:
        raise ValueError("pool_limit must be at least 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")
    if count < 2:
        raise ValueError("count must be at least 2 so train and validation are non-empty")
    if not families:
        raise ValueError("at least one KILT program family is required")
    unknown_families = sorted(set(families) - set(DEFAULT_KILT_FAMILIES))
    if unknown_families:
        raise ValueError(f"unsupported KILT program families: {', '.join(unknown_families)}")
    attempt_limit = max_attempts if max_attempts is not None else count * 50
    if attempt_limit < count:
        raise ValueError("max_attempts must be at least count")

    backend = backend_from_snapshot(snapshot)
    close = getattr(backend, "close", None)
    tasks: list[TaskCertificate] = []
    traces: dict[str, dict[str, Any]] = {}
    rejections: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    family_attempts: Counter[str] = Counter()
    family_accepted: Counter[str] = Counter()
    signatures: set[str] = set()
    rng = random.Random(seed)
    attempts = 0
    try:
        roots = _eligible_roots(
            backend,
            pool_limit=pool_limit,
            min_degree=min_degree,
            max_degree=max_degree,
            rng=rng,
        )
        if not roots:
            raise ValueError("no KILT roots satisfy the configured degree bounds")
        progress = ProgressLogger("data.bootstrap_kilt_grpo.certify", total=attempt_limit)
        progress.start(requested=count, roots=len(roots), seed=seed)
        for attempt in range(attempt_limit):
            attempts = attempt + 1
            root_id = rng.choice(roots)
            family = rng.choice(families)
            family_attempts[family] += 1
            program, construction_reason = _candidate_program(backend, root_id, family, rng)
            codes: tuple[str, ...]
            if program is None:
                codes = (construction_reason or "CANDIDATE_CONSTRUCTION_FAILED",)
                detail = codes[0]
            else:
                signature = canonical_signature(program)
                if signature in signatures:
                    codes = ("DUPLICATE_SIGNATURE",)
                    detail = signature
                else:
                    signatures.add(signature)
                    try:
                        task, trace = _certified_task(
                            backend,
                            program,
                            root_id=root_id,
                            family=family,
                            attempt=attempt,
                            seed=seed,
                            snapshot=snapshot,
                        )
                    except (TraceCompilationError, TypeError, ValueError, KeyError) as exc:
                        codes = _reason_codes(exc)
                        detail = str(exc)
                    else:
                        tasks.append(task)
                        traces[task.task_id] = trace
                        family_accepted[family] += 1
                        progress.update(attempts, accepted=len(tasks), rejected=len(rejections))
                        if len(tasks) >= count:
                            break
                        continue
            for code in codes:
                reason_counts[code] += 1
                rejections.append(
                    {
                        "index": attempt,
                        "reason_code": code,
                        "detail": detail,
                        "family": family,
                        "root_entity_id": root_id,
                        "program": program_to_dict(program) if program is not None else None,
                    }
                )
            progress.update(attempts, accepted=len(tasks), rejected=len(rejections))
        progress.finish(attempts, accepted=len(tasks), rejected=len(rejections))

        if len(tasks) < count:
            write_records(output_dir / "rejections.parquet", rejections)
            failure = {
                "dataset": "kilt",
                "snapshot": snapshot,
                "bootstrap_version": KILT_BOOTSTRAP_VERSION,
                "requested": count,
                "accepted": len(tasks),
                "attempts": attempts,
                "complete": False,
                "seed": seed,
                "eligible_roots": len(roots),
                "rejection_reasons": dict(sorted(reason_counts.items())),
            }
            write_json(output_dir / "metrics.json", failure)
            raise RuntimeError(
                f"KILT bootstrap accepted {len(tasks)} of {count} requested tasks; "
                "increase max_attempts or relax bounded seed filters"
            )

        splits = _split_tasks(tasks, traces, val_ratio=val_ratio, seed=seed)
        relation_ids = sorted(
            {relation for task in tasks for relation in program_relations(task.program)}
        )
        relation_catalog = tuple(backend.relation_info(value) for value in relation_ids)
        split_counts: dict[str, int] = {}
        for split, (split_tasks, split_traces) in splits.items():
            split_counts[split] = len(split_tasks)
            write_records(
                output_dir / split / "tasks.parquet",
                (task.model_dump(mode="json") for task in split_tasks),
            )
            write_records(output_dir / split / "traces.parquet", split_traces)
            if split_tasks:
                export_role_dataset(
                    split_tasks,
                    output_dir / split / "solver_grpo.parquet",
                    include_questioner=False,
                    include_solver=True,
                    interaction_mode=interaction_mode,
                    graphscript_version=graphscript_version,
                    relation_catalog=relation_catalog,
                )
        write_records(output_dir / "rejections.parquet", rejections)
        write_json(
            output_dir / "relation_catalog.json",
            [relation.model_dump(mode="json") for relation in relation_catalog],
        )
        summary = {
            "dataset": "kilt",
            "snapshot": snapshot,
            "bootstrap_version": KILT_BOOTSTRAP_VERSION,
            "requested": count,
            "accepted": len(tasks),
            "attempts": attempts,
            "complete": len(tasks) == count,
            "acceptance_rate": len(tasks) / attempts if attempts else 0.0,
            "splits": split_counts,
            "seed": seed,
            "pool_limit": pool_limit,
            "eligible_roots": len(roots),
            "min_degree": min_degree,
            "max_degree": max_degree,
            "val_ratio": val_ratio,
            "families": list(families),
            "interaction_mode": interaction_mode,
            "graphscript_version": graphscript_version,
            "family_attempts": dict(sorted(family_attempts.items())),
            "family_accepted": dict(sorted(family_accepted.items())),
            "rejection_reasons": dict(sorted(reason_counts.items())),
        }
        write_json(output_dir / "metrics.json", summary)
        write_manifest(
            output_dir,
            {
                "command": "data bootstrap-kilt-grpo",
                "snapshot": snapshot,
                "count": count,
                "seed": seed,
                "pool_limit": pool_limit,
                "max_attempts": attempt_limit,
                "min_degree": min_degree,
                "max_degree": max_degree,
                "val_ratio": val_ratio,
                "families": list(families),
                "interaction_mode": interaction_mode,
                "graphscript_version": graphscript_version,
            },
            [
                "train/tasks.parquet",
                "train/traces.parquet",
                "train/solver_grpo.parquet",
                "val/tasks.parquet",
                "val/traces.parquet",
                "val/solver_grpo.parquet",
                "rejections.parquet",
                "relation_catalog.json",
                "metrics.json",
            ],
        )
        return summary
    finally:
        if callable(close):
            close()
