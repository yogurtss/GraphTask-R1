from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from graphtask_r1.dsl import canonical_signature
from graphtask_r1.evaluation.answer_metrics import answer_metrics
from graphtask_r1.evaluation.kqapro_val import (
    CompletionClient,
    KQAProModelConfig,
    OpenAICompletionClient,
)
from graphtask_r1.graph import GraphBackend
from graphtask_r1.graph.sqlite import SQLiteGraphBackend
from graphtask_r1.graphscript import (
    GraphScriptError,
    execute_graphscript,
    parse_graphscript,
    program_to_graphscript,
)
from graphtask_r1.schema import AnswerSet, RelationInfo, TaskTrainingRecord
from graphtask_r1.training.prompts import role_prompt
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.training.response_normalization import (
    normalize_graphscript_response,
    normalize_reward_response,
)
from graphtask_r1.training.sft_dataset import (
    observable_entity_reference,
    observable_entity_set_reference,
)
from graphtask_r1.utils import file_hash, iter_record_json, stable_hash, write_json

TARGET_CONTRACT = (
    "observable_exact_label_or_complete_same_label_candidate_set_no_hidden_entity_id"
)


@dataclass(frozen=True)
class GRRConfig:
    repo_root: Path
    seed: int
    tasks: Path
    relation_catalog: Path
    graph_db: Path
    model_artifact: Path
    output_dir: Path
    model_url: str
    model_name: str
    max_completion_tokens: int
    enable_thinking: bool
    probe_count: int
    pair_count: int
    max_pairs_per_boundary_family: int | None
    concurrency: int
    request_timeout_s: float
    request_retries: int
    max_follow_limit: int
    max_edge_visits: int
    max_returned_entities: int

    @property
    def rollouts_path(self) -> Path:
        return self.output_dir / "rollouts.jsonl"

    @property
    def rollout_manifest_path(self) -> Path:
        return self.output_dir / "rollout_manifest.json"

    @property
    def pairs_dir(self) -> Path:
        return self.output_dir / "pairs"


def _path(root: Path, value: object) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def load_config(path: Path) -> GRRConfig:
    config_path = path.resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = _path(config_path.parent, raw.get("repo_root", "../../.."))
    paths = dict(raw["paths"])
    model = dict(raw["model"])
    rollout = dict(raw["rollout"])
    pairs = dict(raw["pairs"])
    execution = dict(raw["execution"])
    return GRRConfig(
        repo_root=repo_root,
        seed=int(raw["seed"]),
        tasks=_path(repo_root, paths["tasks"]),
        relation_catalog=_path(repo_root, paths["relation_catalog"]),
        graph_db=_path(repo_root, paths["graph_db"]),
        model_artifact=_path(repo_root, paths["model_artifact"]),
        output_dir=_path(repo_root, paths["output_dir"]),
        model_url=os.path.expandvars(str(model["url"])),
        model_name=os.path.expandvars(str(model["name"])),
        max_completion_tokens=int(model["max_completion_tokens"]),
        enable_thinking=bool(model["enable_thinking"]),
        probe_count=int(rollout["probe_count"]),
        pair_count=int(pairs["count"]),
        max_pairs_per_boundary_family=(
            int(pairs["max_per_boundary_family"])
            if pairs.get("max_per_boundary_family") is not None
            else None
        ),
        concurrency=int(rollout["concurrency"]),
        request_timeout_s=float(rollout["request_timeout_s"]),
        request_retries=int(rollout["request_retries"]),
        max_follow_limit=int(execution["max_follow_limit"]),
        max_edge_visits=int(execution["max_edge_visits"]),
        max_returned_entities=int(execution["max_returned_entities"]),
    )


@dataclass(frozen=True)
class CertifiedTask:
    rank: int
    task: TaskTrainingRecord
    chosen_graphscript: str
    chosen_program_signature: str
    chosen_answers: tuple[str, ...]


def _answer_keys(answers: AnswerSet) -> tuple[str, ...]:
    return tuple(f"{answer.kind}:{answer.value}" for answer in answers.answers)


def _load_probe_tasks(config: GRRConfig) -> list[TaskTrainingRecord]:
    tasks = [
        TaskTrainingRecord.model_validate_json(raw)
        for raw in iter_record_json(config.tasks)
    ]
    return sorted(
        tasks,
        key=lambda task: stable_hash([config.seed, "grr-probe", task.task_id]),
    )[: config.probe_count]


def _certify_task(
    task: TaskTrainingRecord,
    *,
    rank: int,
    backend: GraphBackend,
    relations: tuple[RelationInfo, ...],
    config: GRRConfig,
) -> tuple[CertifiedTask | None, str]:
    entity_references: dict[
        str, tuple[str, Literal["id", "exact", "search"]]
    ] = {}

    def observable_reference(
        entity_id: str,
    ) -> tuple[str, Literal["id", "exact", "search"]]:
        reference = observable_entity_reference(backend, entity_id)
        entity_references[entity_id] = reference
        return reference

    def observable_set_reference(
        entity_ids: tuple[str, ...],
    ) -> tuple[str, Literal["exact"]] | None:
        return observable_entity_set_reference(backend, entity_ids)

    try:
        chosen = program_to_graphscript(
            task.program,
            version="0.3",
            entity_reference=observable_reference,
            entity_set_reference=observable_set_reference,
        )
        if any(match != "exact" for _, match in entity_references.values()):
            return None, "CHOSEN_UNOBSERVABLE_ENTITY_ID"
        execution = execute_graphscript(
            chosen,
            backend,
            allowed_relations=frozenset(value.relation_id for value in relations),
            max_edge_visits=config.max_edge_visits,
            max_returned_entities=config.max_returned_entities,
            trace_id=f"grr:gold:{task.task_id}",
        )
    except GraphScriptError as exc:
        return None, f"CHOSEN_{exc.reason_code}"
    if canonical_signature(task.program) != canonical_signature(execution.program):
        return None, "CHOSEN_PROGRAM_MISMATCH"
    if execution.answers != task.gold_answers:
        return None, "CHOSEN_GOLD_MISMATCH"
    return (
        CertifiedTask(
            rank=rank,
            task=task,
            chosen_graphscript=chosen.model_dump_json(by_alias=True),
            chosen_program_signature=canonical_signature(execution.program),
            chosen_answers=_answer_keys(execution.answers),
        ),
        "CERTIFIED",
    )


def _classify_response(
    example: CertifiedTask,
    response: str,
    *,
    backend: GraphBackend,
    relations: tuple[RelationInfo, ...],
    config: GRRConfig,
) -> dict[str, object]:
    rejected_response = normalize_reward_response(response)
    parser_input = normalize_graphscript_response(response)
    try:
        predicted = parse_graphscript(
            parser_input,
            max_follow_limit=config.max_follow_limit,
        )
    except GraphScriptError as exc:
        return {
            "failure_stage": "predicted_parse",
            "reason_code": exc.reason_code,
            "tool_succeeded": False,
            "terminal_f1": 0.0,
            "terminal_exact_match": 0.0,
            "rejected_response": rejected_response,
        }
    try:
        execution = execute_graphscript(
            predicted,
            backend,
            allowed_relations=frozenset(value.relation_id for value in relations),
            max_edge_visits=config.max_edge_visits,
            max_returned_entities=config.max_returned_entities,
            trace_id=f"grr:predicted:{example.task.task_id}",
        )
    except GraphScriptError as exc:
        return {
            "failure_stage": "predicted_execution",
            "reason_code": exc.reason_code,
            "tool_succeeded": False,
            "terminal_f1": 0.0,
            "terminal_exact_match": 0.0,
            "rejected_response": rejected_response,
            "predicted_graphscript": predicted.model_dump(mode="json", by_alias=True),
        }
    metrics = answer_metrics(execution.answers, example.task.gold_answers)
    exact = float(metrics["exact_match"])
    return {
        "failure_stage": "terminal_semantic",
        "reason_code": "TERMINAL_CORRECT" if exact == 1.0 else "TOOL_WRONG_ANSWER",
        "tool_succeeded": True,
        "terminal_f1": float(metrics["f1"]),
        "terminal_exact_match": exact,
        "rejected_response": rejected_response,
        "predicted_graphscript": predicted.model_dump(mode="json", by_alias=True),
        "predicted_program_signature": canonical_signature(execution.program),
        "predicted_answers": list(_answer_keys(execution.answers)),
    }


async def collect_fixed_incumbent_rollouts(
    config: GRRConfig,
    *,
    backend: GraphBackend | None = None,
    client: CompletionClient | None = None,
) -> dict[str, object]:
    """Collect one greedy, task-local response from one fixed incumbent snapshot."""

    effective_backend = backend or SQLiteGraphBackend(
        config.graph_db,
        snapshot_id="kqapro-v1",
    )
    relations = load_relation_catalog(config.relation_catalog)
    probe_tasks = _load_probe_tasks(config)
    certification_reasons: Counter[str] = Counter()
    certified: list[CertifiedTask] = []
    for rank, task in enumerate(probe_tasks):
        example, reason = _certify_task(
            task,
            rank=rank,
            backend=effective_backend,
            relations=relations,
            config=config,
        )
        certification_reasons[reason] += 1
        if example is not None:
            certified.append(example)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    owned_client = client is None
    effective_client = client or OpenAICompletionClient(
        KQAProModelConfig(
            model_url=config.model_url,
            model=config.model_name,
            max_completion_tokens=config.max_completion_tokens,
            chat_template_kwargs={"enable_thinking": config.enable_thinking},
        ),
        timeout_s=config.request_timeout_s,
        retries=config.request_retries,
        cache_path=config.output_dir / "cache" / "rollouts.json",
    )
    semaphore = asyncio.Semaphore(config.concurrency)

    async def collect(example: CertifiedTask) -> tuple[int, dict[str, object]]:
        async with semaphore:
            messages = role_prompt(
                "solver",
                f"Question: {example.task.question}",
                interaction_mode="graphscript",
                relation_catalog=relations,
                graphscript_version="0.3",
            )
            sample_seed = config.seed + example.rank
            completion = await effective_client.complete(
                messages,
                trace_id=(
                    f"grr-rollout:{config.model_name}:{example.task.task_id}:"
                    f"{sample_seed}"
                ),
                seed=sample_seed,
            )
            classification = _classify_response(
                example,
                completion.content,
                backend=effective_backend,
                relations=relations,
                config=config,
            )
            record: dict[str, object] = {
                "algorithm": "graph_certified_on_policy_residual_replay_v1",
                "task_id": example.task.task_id,
                "task_rank": example.rank,
                "sample_seed": sample_seed,
                "question": example.task.question,
                "chosen_graphscript": example.chosen_graphscript,
                "chosen_program_signature": example.chosen_program_signature,
                "chosen_answers": list(example.chosen_answers),
                "response_sha256": stable_hash(completion.content),
                "completion_tokens": completion.completion_tokens,
                "cache_hit": completion.cached,
                **classification,
            }
            return example.rank, record

    try:
        results = await asyncio.gather(*(collect(example) for example in certified))
    finally:
        if owned_client:
            effective_client.flush()  # type: ignore[attr-defined]
    records = [record for _, record in sorted(results)]
    with config.rollouts_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    reason_counts = Counter(str(record["reason_code"]) for record in records)
    stage_counts = Counter(str(record["failure_stage"]) for record in records)
    manifest: dict[str, object] = {
        "algorithm": "graph_certified_on_policy_residual_replay_v1",
        "generation_mode": "fixed_incumbent_greedy_task_local",
        "target_contract": TARGET_CONTRACT,
        "seed": config.seed,
        "source_tasks": str(config.tasks),
        "source_tasks_sha256": file_hash(config.tasks),
        "probe_count": config.probe_count,
        "probe_task_ids": [task.task_id for task in probe_tasks],
        "certified_probe_count": len(certified),
        "certification_reason_counts": dict(sorted(certification_reasons.items())),
        "rollout_count": len(records),
        "incorrect_rollout_count": sum(
            record["reason_code"] != "TERMINAL_CORRECT" for record in records
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "model_name": config.model_name,
        "model_artifact": str(config.model_artifact),
        "model_artifact_sha256": file_hash(config.model_artifact),
        "temperature": 0.0,
        "max_completion_tokens": config.max_completion_tokens,
        "enable_thinking": config.enable_thinking,
        "relation_catalog": str(config.relation_catalog),
        "relation_catalog_sha256": file_hash(config.relation_catalog),
        "graph_db": str(config.graph_db),
        "graph_db_sha256": file_hash(config.graph_db),
        "max_edge_visits": config.max_edge_visits,
        "max_returned_entities": config.max_returned_entities,
        "rollouts": str(config.rollouts_path),
        "rollouts_sha256": file_hash(config.rollouts_path),
    }
    write_json(config.rollout_manifest_path, manifest)
    return manifest


def build_replay_pairs(config: GRRConfig) -> dict[str, object]:
    """Pair each fixed-incumbent failure with its same-task certified program."""

    rollout_manifest = json.loads(config.rollout_manifest_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in config.rollouts_path.read_text(encoding="utf-8").splitlines()
    ]
    failures = [
        record
        for record in records
        if record["reason_code"] != "TERMINAL_CORRECT"
        and str(record["rejected_response"])
    ]
    failures.sort(key=lambda record: int(record["task_rank"]))
    selected = failures[: config.pair_count]
    relations = load_relation_catalog(config.relation_catalog)
    config.pairs_dir.mkdir(parents=True, exist_ok=True)
    train_path = config.pairs_dir / "train.jsonl"
    certificate_path = config.pairs_dir / "residual_certificates.jsonl"
    stage_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    with (
        train_path.open("w", encoding="utf-8") as train_stream,
        certificate_path.open("w", encoding="utf-8") as certificate_stream,
    ):
        for record in selected:
            messages = role_prompt(
                "solver",
                f"Question: {record['question']}",
                interaction_mode="graphscript",
                relation_catalog=relations,
                graphscript_version="0.3",
            )
            messages.append(
                {"role": "assistant", "content": str(record["chosen_graphscript"])}
            )
            pair = {
                "messages": messages,
                "rejected_response": str(record["rejected_response"]),
                "task_id": str(record["task_id"]),
                "grr_failure_stage": str(record["failure_stage"]),
                "grr_reason_code": str(record["reason_code"]),
            }
            certificate = {
                "task_id": str(record["task_id"]),
                "task_rank": int(record["task_rank"]),
                "sample_seed": int(record["sample_seed"]),
                "failure_stage": str(record["failure_stage"]),
                "reason_code": str(record["reason_code"]),
                "terminal_f1": float(record["terminal_f1"]),
                "terminal_exact_match": float(record["terminal_exact_match"]),
                "tool_succeeded": bool(record["tool_succeeded"]),
                "chosen_program_signature": str(record["chosen_program_signature"]),
                "chosen_answers": list(record["chosen_answers"]),
                "rejected_response_sha256": stable_hash(record["rejected_response"]),
                "response_sha256": str(record["response_sha256"]),
                "same_task": True,
                "chosen_graph_certified": True,
                "rejected_sampled_from_fixed_incumbent": True,
            }
            train_stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
            certificate_stream.write(
                json.dumps(certificate, ensure_ascii=False, sort_keys=True) + "\n"
            )
            stage_counts[str(certificate["failure_stage"])] += 1
            reason_counts[str(certificate["reason_code"])] += 1

    manifest: dict[str, object] = {
        "algorithm": "graph_certified_on_policy_residual_replay_v1",
        "generation_mode": "task_local_fixed_incumbent_error_replay",
        "seed": config.seed,
        "requested_pairs": config.pair_count,
        "selected_pairs": len(selected),
        "selected_task_ids": [str(record["task_id"]) for record in selected],
        "stage_counts": dict(sorted(stage_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rollout_manifest": str(config.rollout_manifest_path),
        "rollout_manifest_sha256": file_hash(config.rollout_manifest_path),
        "rollouts_sha256": file_hash(config.rollouts_path),
        "model_artifact_sha256": rollout_manifest["model_artifact_sha256"],
        "train": str(train_path),
        "train_sha256": file_hash(train_path),
        "certificates": str(certificate_path),
        "certificates_sha256": file_hash(certificate_path),
    }
    write_json(config.pairs_dir / "manifest.json", manifest)
    return manifest
