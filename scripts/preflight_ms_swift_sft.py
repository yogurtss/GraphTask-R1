#!/usr/bin/env python3
"""Split an SFT Parquet with the exact ms-swift template/token-length policy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.training.ms_swift_data import convert_sft_row
from graphtask_r1.utils import write_json

HARD_MAX_LENGTH = 40_960
_LENGTH_PATTERN = re.compile(r"Current length of row\((\d+)\)")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GraphTask SFT rows with ms-swift without starting training."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", default="qwen3")
    parser.add_argument("--template", default="qwen3")
    parser.add_argument("--agent-template", default="hermes")
    parser.add_argument("--max-length", type=int, default=32_768)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not 1 <= args.max_length <= HARD_MAX_LENGTH:
        raise ValueError(f"max-length must be between 1 and {HARD_MAX_LENGTH}")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    try:
        from swift.llm import MaxLengthError, get_model_tokenizer, get_template
    except ImportError as exc:
        raise RuntimeError("install ms-swift==3.6.4 before running this preflight") from exc

    _, processor = get_model_tokenizer(
        str(args.model),
        load_model=False,
        model_type=args.model_type,
    )
    template = get_template(
        args.template,
        processor,
        max_length=args.max_length,
        truncation_strategy="raise",
        agent_template=args.agent_template,
    )
    template.set_mode("train")

    table = pq.read_table(args.input)
    accepted_indices: list[int] = []
    rejections: list[dict[str, object]] = []
    lengths: list[int] = []
    for index, row in enumerate(table.to_pylist()):
        try:
            encoded = template.encode(convert_sft_row(row))
            length = int(encoded.get("length", len(encoded["input_ids"])))
            lengths.append(length)
            accepted_indices.append(index)
        except MaxLengthError as exc:
            match = _LENGTH_PATTERN.search(str(exc))
            rejections.append(
                {
                    "row_index": index,
                    "task_id": str(row.get("task_id", "")),
                    "role": str(row.get("role", "")),
                    "token_length": int(match.group(1)) if match else None,
                    "max_length": args.max_length,
                    "reason_code": "SFT_MAX_LENGTH_EXCEEDED",
                    "detail": str(exc),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rejections.append(
                {
                    "row_index": index,
                    "task_id": str(row.get("task_id", "")),
                    "role": str(row.get("role", "")),
                    "token_length": None,
                    "max_length": args.max_length,
                    "reason_code": "SFT_ENCODING_ERROR",
                    "detail": str(exc),
                }
            )

    args.accepted_output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_output.parent.mkdir(parents=True, exist_ok=True)
    accepted = table.take(pa.array(accepted_indices, type=pa.int64()))
    pq.write_table(accepted, args.accepted_output)
    rejection_schema = pa.schema(
        [
            ("row_index", pa.int64()),
            ("task_id", pa.string()),
            ("role", pa.string()),
            ("token_length", pa.int64()),
            ("max_length", pa.int64()),
            ("reason_code", pa.string()),
            ("detail", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rejections, schema=rejection_schema), args.rejected_output)
    summary = {
        "input": str(args.input),
        "accepted_output": str(args.accepted_output),
        "rejected_output": str(args.rejected_output),
        "max_length": args.max_length,
        "input_rows": table.num_rows,
        "accepted_rows": len(accepted_indices),
        "rejected_rows": len(rejections),
        "accepted_min_tokens": min(lengths) if lengths else None,
        "accepted_max_tokens": max(lengths) if lengths else None,
    }
    if args.summary_output is not None:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if args.require_all and rejections else 0


if __name__ == "__main__":
    raise SystemExit(main())
