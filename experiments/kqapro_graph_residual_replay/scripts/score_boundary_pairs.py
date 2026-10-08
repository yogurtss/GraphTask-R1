#!/usr/bin/env python3
"""Select current-policy-confusing execution boundaries under one canonical prefix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

import kqapro_grr.ms_swift_boundary_plugin  # noqa: F401
import torch
import torch.nn.functional as functional
from swift.llm import get_model_tokenizer, get_template
from transformers import AutoModelForCausalLM

from graphtask_r1.utils import file_hash, write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _boundary_encoded(template: Any, row: dict[str, Any]) -> dict[str, Any]:
    boundary_row = deepcopy(row)
    boundary_row["messages"][-1].pop("loss")
    return cast(dict[str, Any], template.encode(boundary_row))


def _mean_conditional_log_prob(
    model: Any,
    input_ids: list[list[int]],
    labels: list[list[int]],
    *,
    pad_token_id: int,
    device: str,
) -> tuple[float, float, int, int]:
    max_length = max(len(value) for value in input_ids)
    batch_ids = torch.full(
        (len(input_ids), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(batch_ids)
    for row_index, values in enumerate(input_ids):
        length = len(values)
        batch_ids[row_index, :length] = torch.tensor(values, device=device)
        attention_mask[row_index, :length] = 1

    with torch.inference_mode():
        hidden = model.model(
            input_ids=batch_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        scores: list[float] = []
        counts: list[int] = []
        for row_index, row_labels in enumerate(labels):
            positions = [
                position
                for position in range(1, len(row_labels))
                if row_labels[position] != -100
            ]
            position_tensor = torch.tensor(positions, device=device)
            logits = model.lm_head(hidden[row_index, position_tensor - 1]).float()
            targets = batch_ids[row_index, position_tensor]
            log_probs = -functional.cross_entropy(logits, targets, reduction="none")
            scores.append(float(log_probs.mean().item()))
            counts.append(len(positions))
    return scores[0], scores[1], counts[0], counts[1]


def select_family_balanced_hard_boundaries(
    scores: list[dict[str, object]],
    *,
    keep_total: int,
    keep_per_family: int,
) -> list[dict[str, object]]:
    def confusion_margin(value: dict[str, object]) -> float:
        return cast(float, value["confusion_margin"])

    hard_scores = [score for score in scores if confusion_margin(score) > 0.0]
    ranked_by_family: dict[str, list[dict[str, object]]] = {}
    for score in hard_scores:
        ranked_by_family.setdefault(str(score["boundary_family"]), []).append(score)
    if not ranked_by_family:
        return []
    for family_scores in ranked_by_family.values():
        family_scores.sort(
            key=lambda value: (-confusion_margin(value), str(value["task_id"]))
        )

    base_quota = keep_total // len(ranked_by_family)
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    for family in sorted(ranked_by_family):
        for score in ranked_by_family[family][: min(base_quota, keep_per_family)]:
            selected.append(score)
            selected_ids.add(str(score["task_id"]))
            family_counts[family] += 1

    remainder = sorted(
        (
            score
            for score in hard_scores
            if str(score["task_id"]) not in selected_ids
        ),
        key=lambda value: (-confusion_margin(value), str(value["task_id"])),
    )
    for score in remainder:
        family = str(score["boundary_family"])
        if family_counts[family] >= keep_per_family:
            continue
        selected.append(score)
        family_counts[family] += 1
        if len(selected) == keep_total:
            break
    selected.sort(
        key=lambda value: (-confusion_margin(value), str(value["task_id"]))
    )
    return selected


def score_and_select(
    *,
    dataset: Path,
    control_dataset: Path,
    source_manifest: Path,
    model_path: Path,
    output_dir: Path,
    keep_total: int,
    keep_per_family: int,
    device: str,
) -> dict[str, object]:
    rows = _read_jsonl(dataset)
    if len(rows) <= keep_total:
        raise ValueError(
            f"hardness selection needs more than {keep_total} candidates; got {len(rows)}"
        )
    control_by_id = {
        str(row["task_id"]): row for row in _read_jsonl(control_dataset)
    }
    _, tokenizer = get_model_tokenizer(
        str(model_path),
        load_model=False,
        model_type="qwen3",
    )
    template = get_template(
        "qwen3_execution_boundary",
        tokenizer,
        max_length=32768,
        loss_scale="execution_boundary_asymmetric",
        response_prefix="<think>\n\n</think>\n\n",
    )
    template.set_mode("rlhf")
    model = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model.eval()

    scores: list[dict[str, object]] = []
    for row in rows:
        encoded = _boundary_encoded(template, row)
        chosen_score, rejected_score, chosen_tokens, rejected_tokens = (
            _mean_conditional_log_prob(
                model,
                [encoded["chosen_input_ids"], encoded["rejected_input_ids"]],
                [encoded["chosen_labels"], encoded["rejected_labels"]],
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
        )
        scores.append(
            {
                "task_id": str(row["task_id"]),
                "boundary_family": str(row["boundary_family"]),
                "chosen_mean_log_prob": chosen_score,
                "rejected_mean_log_prob": rejected_score,
                "confusion_margin": rejected_score - chosen_score,
                "chosen_boundary_tokens": chosen_tokens,
                "rejected_boundary_tokens": rejected_tokens,
            }
        )

    selected_scores = select_family_balanced_hard_boundaries(
        scores,
        keep_total=keep_total,
        keep_per_family=keep_per_family,
    )
    if len(selected_scores) != keep_total:
        raise ValueError(
            f"only {len(selected_scores)} family-balanced hard boundaries for {keep_total} slots"
        )
    selected_ids = {str(value["task_id"]) for value in selected_scores}
    row_by_id = {str(row["task_id"]): row for row in rows}
    selected_rows = [row_by_id[str(value["task_id"])] for value in selected_scores]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    control_path = output_dir / "train_control.jsonl"
    score_path = output_dir / "hardness_scores.jsonl"
    with (
        train_path.open("w", encoding="utf-8") as train_stream,
        control_path.open("w", encoding="utf-8") as control_stream,
        score_path.open("w", encoding="utf-8") as score_stream,
    ):
        for row in selected_rows:
            task_id = str(row["task_id"])
            train_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            control_stream.write(
                json.dumps(control_by_id[task_id], ensure_ascii=False) + "\n"
            )
        for score in sorted(scores, key=lambda value: str(value["task_id"])):
            score_stream.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")

    margins = [cast(float, value["confusion_margin"]) for value in scores]
    selected_margins = [
        cast(float, value["confusion_margin"])
        for value in scores
        if str(value["task_id"]) in selected_ids
    ]
    manifest: dict[str, object] = {
        "algorithm": "incumbent_same_prefix_boundary_confusion_selection_v1",
        "score": "mean_token_logp_rejected_minus_mean_token_logp_chosen",
        "source_dataset": str(dataset),
        "source_dataset_sha256": file_hash(dataset),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_hash(source_manifest),
        "model": str(model_path),
        "model_artifact_sha256": file_hash(model_path / "model.safetensors"),
        "candidate_count": len(rows),
        "positive_margin_candidate_count": sum(margin > 0.0 for margin in margins),
        "keep_total": keep_total,
        "keep_per_family": keep_per_family,
        "selected_count": len(selected_rows),
        "candidate_family_counts": dict(
            sorted(Counter(str(row["boundary_family"]) for row in rows).items())
        ),
        "selected_family_counts": dict(
            sorted(Counter(str(row["boundary_family"]) for row in selected_rows).items())
        ),
        "candidate_margin_mean": mean(margins),
        "candidate_margin_median": median(margins),
        "selected_margin_mean": mean(selected_margins),
        "selected_margin_median": median(selected_margins),
        "train": str(train_path),
        "train_sha256": file_hash(train_path),
        "control_train": str(control_path),
        "control_train_sha256": file_hash(control_path),
        "scores": str(score_path),
        "scores_sha256": file_hash(score_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--control-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-total", type=int, default=64)
    parser.add_argument("--keep-per-family", type=int, default=27)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = score_and_select(
        dataset=args.dataset,
        control_dataset=args.control_dataset,
        source_manifest=args.source_manifest,
        model_path=args.model,
        output_dir=args.output_dir,
        keep_total=args.keep_total,
        keep_per_family=args.keep_per_family,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
