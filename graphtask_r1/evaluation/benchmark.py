from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

from graphtask_r1.schema import BenchmarkExample
from graphtask_r1.training.prompts import GraphScriptVersion
from graphtask_r1.utils import read_records, write_json, write_records


async def evaluate_benchmark(
    input_path: Path,
    output_dir: Path,
    *,
    solver_url: str,
    graph_snapshot: str = "freebase-v1",
    samples: int = 1,
    concurrency: int = 16,
    graphscript_version: GraphScriptVersion = "0.2",
) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - training extra
        raise ImportError("install aiohttp from requirements.txt for benchmark evaluation") from exc
    examples = [BenchmarkExample.model_validate(value) for value in read_records(input_path)]
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_one(example: BenchmarkExample) -> dict[str, Any]:
        async with semaphore:
            payload = {
                "example": example.model_dump(mode="json"),
                "graph_snapshot": graph_snapshot,
                "samples": samples,
                "graphscript_version": graphscript_version,
            }
            timeout = aiohttp.ClientTimeout(total=300)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(solver_url.rstrip("/") + "/solve", json=payload) as response,
            ):
                body = await response.json()
                if response.status != 200:
                    raise RuntimeError(f"solver service failed: {body}")
                return {**body, "split": example.split, "dataset": example.dataset}

    results = await asyncio.gather(*(evaluate_one(example) for example in examples))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        buckets[str(result["split"])].append(result)

    def summarize(values: list[dict[str, Any]]) -> dict[str, float | int]:
        count = len(values)
        return {
            "count": count,
            "entity_f1": sum(float(value["mean_f1"]) for value in values) / count,
            "exact_match": sum(float(value["pass_rate"]) for value in values) / count,
            "mean_tool_calls": sum(float(value["mean_tool_calls"]) for value in values) / count,
            "mean_edge_visits": sum(float(value.get("mean_edge_visits", 0.0)) for value in values)
            / count,
            "mean_latency_ms": sum(float(value["mean_latency_ms"]) for value in values) / count,
            "program_parse_rate": sum(
                float(value.get("program_parse_rate", 0.0)) for value in values
            )
            / count,
            "program_execution_rate": sum(
                float(value.get("program_execution_rate", 0.0)) for value in values
            )
            / count,
            "mean_program_operators": sum(
                float(value.get("mean_program_operators", 0.0)) for value in values
            )
            / count,
            "mean_passage_searches": sum(
                float(value.get("mean_passage_searches", 0.0)) for value in values
            )
            / count,
        }

    summary = {
        "dataset": examples[0].dataset if examples else "unknown",
        "input": str(input_path),
        "samples": samples,
        "graphscript_version": graphscript_version,
        "overall": summarize(results) if results else {},
        "by_split": {split: summarize(values) for split, values in buckets.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(output_dir / "predictions.parquet", results)
    write_json(output_dir / "metrics.json", summary)
    return summary
