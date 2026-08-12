#!/usr/bin/env python3
"""Split an SFT Parquet with the exact ms-swift template/token-length policy."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.training.ms_swift_data import convert_sft_row
from graphtask_r1.utils import ParquetRowWriter, ProgressLogger, write_json

HARD_MAX_LENGTH = 40_960
_LENGTH_PATTERN = re.compile(r"Current length of row\((\d+)\)")
LOGGER = logging.getLogger("graphtask_r1.preflight_ms_swift_sft")


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.setLevel(logging.INFO)
    if not 1 <= args.max_length <= HARD_MAX_LENGTH:
        raise ValueError(f"max-length must be between 1 and {HARD_MAX_LENGTH}")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    try:
        from swift.llm import MaxLengthError, get_model_tokenizer, get_template
    except ImportError as exc:
        raise RuntimeError("install ms-swift==3.6.4 before running this preflight") from exc
    # ms-swift installs an ERROR-level root handler during import. Keep the caller's handler,
    # but restore INFO visibility for template loading and row progress.
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    LOGGER.info(
        "template_loading model=%s model_type=%s template=%s agent_template=%s",
        args.model,
        args.model_type,
        args.template,
        args.agent_template,
    )
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
    LOGGER.info("template_loaded model=%s max_length=%d", args.model, args.max_length)

    parquet = pq.ParquetFile(args.input)
    total = int(parquet.metadata.num_rows)
    args.accepted_output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_output.parent.mkdir(parents=True, exist_ok=True)
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
    accepted_writer = pq.ParquetWriter(args.accepted_output, parquet.schema_arrow)
    accepted_rows = 0
    rejected_rows = 0
    min_tokens: int | None = None
    max_tokens: int | None = None
    row_index = 0
    progress_logger = logging.getLogger("graphtask_r1.progress")
    progress_logger.setLevel(logging.INFO)
    progress = ProgressLogger(
        "data.preflight_ms_swift_sft",
        total=total,
        logger=progress_logger,
    )
    progress.start(input=str(args.input), max_length=args.max_length, loading="streaming")
    try:
        with ParquetRowWriter(
            args.rejected_output,
            schema=rejection_schema,
            batch_size=256,
        ) as rejection_writer:
            for batch in parquet.iter_batches(batch_size=64):
                accepted_indices: list[int] = []
                for local_index, row in enumerate(batch.to_pylist()):
                    try:
                        encoded = template.encode(convert_sft_row(row))
                        length = int(encoded.get("length", len(encoded["input_ids"])))
                        min_tokens = length if min_tokens is None else min(min_tokens, length)
                        max_tokens = length if max_tokens is None else max(max_tokens, length)
                        accepted_indices.append(local_index)
                        accepted_rows += 1
                    except MaxLengthError as exc:
                        match = _LENGTH_PATTERN.search(str(exc))
                        rejection_writer.write(
                            {
                                "row_index": row_index,
                                "task_id": str(row.get("task_id", "")),
                                "role": str(row.get("role", "")),
                                "token_length": int(match.group(1)) if match else None,
                                "max_length": args.max_length,
                                "reason_code": "SFT_MAX_LENGTH_EXCEEDED",
                                "detail": str(exc),
                            }
                        )
                        rejected_rows += 1
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        rejection_writer.write(
                            {
                                "row_index": row_index,
                                "task_id": str(row.get("task_id", "")),
                                "role": str(row.get("role", "")),
                                "token_length": None,
                                "max_length": args.max_length,
                                "reason_code": "SFT_ENCODING_ERROR",
                                "detail": str(exc),
                            }
                        )
                        rejected_rows += 1
                    row_index += 1
                    progress.update(
                        row_index,
                        accepted=accepted_rows,
                        rejected=rejected_rows,
                    )
                if accepted_indices:
                    accepted_writer.write_batch(
                        batch.take(pa.array(accepted_indices, type=pa.int32()))
                    )
    finally:
        accepted_writer.close()
    progress.finish(total, accepted=accepted_rows, rejected=rejected_rows)
    summary = {
        "input": str(args.input),
        "accepted_output": str(args.accepted_output),
        "rejected_output": str(args.rejected_output),
        "max_length": args.max_length,
        "input_rows": total,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "accepted_min_tokens": min_tokens,
        "accepted_max_tokens": max_tokens,
    }
    if args.summary_output is not None:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if args.require_all and rejected_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
