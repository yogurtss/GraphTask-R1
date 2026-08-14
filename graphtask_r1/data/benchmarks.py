from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, cast

from graphtask_r1.schema import Answer, AnswerSet, BenchmarkExample
from graphtask_r1.utils import (
    ProgressLogger,
    file_hash,
    ordered_parallel_map,
    validate_workers,
    write_json,
    write_manifest,
    write_records,
)

_MID = re.compile(r"(?:ns:|/ns/|rdf\.freebase\.com/ns/)([mg]\.[A-Za-z0-9_]+)")

OpenQADataset = Literal[
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "bamboogle",
    "musique",
]

COEVOKG_SSP_DATASETS: tuple[OpenQADataset, ...] = (
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "bamboogle",
)

SSP_REVISION = "ce7a0dfbc862f923ad1668a471c409b2e023b73f"
SSP_TEST_SHA256 = "871c7b7cdec2e090e8597ef26a9a973a46aad0830bb1e016679dddd748462f50"
SSP_EXPECTED_COUNTS: dict[OpenQADataset, int] = {
    "nq": 500,
    "triviaqa": 500,
    "popqa": 500,
    "hotpotqa": 500,
    "2wikimultihopqa": 500,
    "bamboogle": 125,
    "musique": 500,
}

_SSP_SOURCE_DATASETS: dict[str, OpenQADataset] = {
    f"searchR1_{dataset}": dataset for dataset in (*COEVOKG_SSP_DATASETS, "musique")
}


def _normalize_id(value: str) -> str:
    match = _MID.search(value)
    return match.group(1) if match else value


def _answers(values: Any) -> AnswerSet:
    if not isinstance(values, list):
        values = [] if values is None else [values]
    answers: list[Answer] = []
    for value in values:
        if isinstance(value, dict):
            raw = (
                value.get("AnswerArgument")
                or value.get("answer_argument")
                or value.get("answer_id")
                or value.get("id")
                or value.get("answer")
                or value.get("EntityName")
            )
            label = value.get("EntityName") or value.get("entity_name") or value.get("label")
        else:
            raw, label = value, None
        if raw is not None:
            normalized = _normalize_id(str(raw))
            kind: Literal["entity", "literal"] = (
                "entity" if normalized.startswith(("m.", "g.")) else "literal"
            )
            answers.append(Answer(value=normalized, kind=kind, label=str(label) if label else None))
    return AnswerSet(answers=tuple(answers))


def _topics(sparql: str | None, explicit: Any = None) -> tuple[str, ...]:
    values: set[str] = set()
    if explicit:
        if isinstance(explicit, str):
            explicit = [explicit]
        values.update(_normalize_id(str(value)) for value in explicit if value)
    if sparql:
        values.update(_MID.findall(sparql))
    return tuple(sorted(value for value in values if value.startswith(("m.", "g."))))


def _load_json_files(raw_dir: Path, *, workers: int) -> list[tuple[Path, Any]]:
    files = sorted([*raw_dir.glob("*.json"), *raw_dir.glob("*.jsonl")])
    loaded: list[tuple[Path, Any]] = []
    progress = ProgressLogger("data.prepare.benchmark.load_files", total=len(files))
    progress.start(raw_dir=str(raw_dir), workers=workers)

    def load(path: Path) -> tuple[Path, Any]:
        if path.suffix == ".jsonl":
            return path, [json.loads(line) for line in path.read_text().splitlines() if line]
        return path, json.loads(path.read_text())

    for index, (path, payload) in enumerate(ordered_parallel_map(load, files, workers=workers)):
        loaded.append((path, payload))
        progress.update(index + 1, file=path.name)
    progress.finish(len(files), raw_dir=str(raw_dir))
    return loaded


def _webqsp_rows(payload: Any, split: str, workers: int) -> list[BenchmarkExample]:
    rows = payload.get("Questions", payload) if isinstance(payload, dict) else payload
    examples: list[BenchmarkExample] = []
    progress = ProgressLogger(f"data.prepare.webqsp.parse.{split}", total=len(rows))
    progress.start(workers=workers)

    def convert(item: tuple[int, dict[str, Any]]) -> BenchmarkExample:
        index, row = item
        parses = row.get("Parses", [])
        parse = next(
            (value for value in parses if value.get("Sparql")), parses[0] if parses else {}
        )
        sparql = parse.get("Sparql")
        all_answers = [answer for value in parses for answer in value.get("Answers", [])]
        all_topics = [value.get("TopicEntityMid") for value in parses]
        return BenchmarkExample(
            example_id=str(row.get("QuestionId", row.get("id", index))),
            dataset="webqsp",
            split=split,
            question=str(row.get("RawQuestion", row.get("ProcessedQuestion", ""))),
            topic_entity_ids=_topics(sparql, all_topics),
            gold_answers=_answers(all_answers or row.get("Answers", [])),
            sparql=sparql,
            metadata={"parse_count": len(parses)},
        )

    for index, example in enumerate(
        ordered_parallel_map(convert, enumerate(rows), workers=workers)
    ):
        examples.append(example)
        progress.update(index + 1)
    progress.finish(len(rows), examples=len(examples))
    return examples


def _cwq_rows(payload: Any, split: str, workers: int) -> list[BenchmarkExample]:
    rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
    examples: list[BenchmarkExample] = []
    progress = ProgressLogger(f"data.prepare.cwq.parse.{split}", total=len(rows))
    progress.start(workers=workers)

    def convert(item: tuple[int, dict[str, Any]]) -> BenchmarkExample:
        index, row = item
        sparql = row.get("sparql") or row.get("SPARQL")
        return BenchmarkExample(
            example_id=str(row.get("ID", row.get("id", index))),
            dataset="cwq",
            split=str(row.get("split", split)),
            question=str(row.get("question", row.get("machine_question", ""))),
            topic_entity_ids=_topics(sparql, row.get("topic_entities")),
            gold_answers=_answers(row.get("answers", row.get("answer", []))),
            logical_form=row.get("s_expression") or row.get("logical_form"),
            sparql=sparql,
        )

    for index, example in enumerate(
        ordered_parallel_map(convert, enumerate(rows), workers=workers)
    ):
        examples.append(example)
        progress.update(index + 1)
    progress.finish(len(rows), examples=len(examples))
    return examples


def _grailqa_rows(payload: Any, split: str, workers: int) -> list[BenchmarkExample]:
    rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
    examples: list[BenchmarkExample] = []
    progress = ProgressLogger(f"data.prepare.grailqa.parse.{split}", total=len(rows))
    progress.start(workers=workers)

    def convert(item: tuple[int, dict[str, Any]]) -> BenchmarkExample:
        index, row = item
        graph_query = row.get("graph_query", {})
        nodes = graph_query.get("nodes", []) if isinstance(graph_query, dict) else []
        explicit = [
            node.get("id")
            for node in nodes
            if node.get("node_type") == "entity" and not node.get("question_node", False)
        ]
        return BenchmarkExample(
            example_id=str(row.get("qid", row.get("id", index))),
            dataset="grailqa",
            split=str(row.get("split", split)),
            question=str(row.get("question", "")),
            topic_entity_ids=_topics(None, explicit),
            gold_answers=_answers(row.get("answer", row.get("answers", []))),
            logical_form=row.get("s_expression") or row.get("logical_form"),
            metadata={"function": row.get("function")},
        )

    for index, example in enumerate(
        ordered_parallel_map(convert, enumerate(rows), workers=workers)
    ):
        examples.append(example)
        progress.update(index + 1)
    progress.finish(len(rows), examples=len(examples))
    return examples


def _alias_answers(values: Any) -> tuple[AnswerSet, tuple[tuple[str, ...], ...]]:
    if not isinstance(values, list | tuple):
        values = [] if values is None else [values]
    aliases = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not aliases:
        return AnswerSet(), ()
    return (
        AnswerSet(answers=(Answer(value=aliases[0], kind="literal"),)),
        (aliases,),
    )


def _provenance_ids(outputs: Any) -> tuple[str, ...]:
    if not isinstance(outputs, list):
        return ()
    values = {
        str(provenance["wikipedia_id"])
        for output in outputs
        if isinstance(output, dict)
        for provenance in output.get("provenance", []) or []
        if isinstance(provenance, dict) and provenance.get("wikipedia_id") is not None
    }
    return tuple(sorted(values))


def _openqa_rows(
    dataset: OpenQADataset, payload: Any, split: str, workers: int
) -> list[BenchmarkExample]:
    if isinstance(payload, dict):
        rows = payload.get("Data", payload.get("data", payload.get("questions", payload)))
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"{dataset} input must contain a JSON array")
    examples: list[BenchmarkExample] = []
    progress = ProgressLogger(f"data.prepare.{dataset}.parse.{split}", total=len(rows))
    progress.start(workers=workers)

    def convert(item: tuple[int, dict[str, Any]]) -> BenchmarkExample:
        index, row = item
        outputs = row.get("output")
        if "input" in row and isinstance(outputs, list):
            answer_values = [
                output.get("answer")
                for output in outputs
                if isinstance(output, dict) and output.get("answer") is not None
            ]
            question = str(row.get("input", ""))
            topic_ids = _provenance_ids(outputs)
            source_format = "kilt"
        elif dataset == "triviaqa" and isinstance(row.get("Answer"), dict):
            answer = row["Answer"]
            answer_values = (
                answer.get("Aliases") or answer.get("NormalizedAliases") or [answer.get("Value")]
            )
            question = str(row.get("Question", row.get("question", "")))
            topic_ids = ()
            source_format = "official"
        else:
            answer_values = row.get("answers", row.get("answer", []))
            question = str(row.get("question", row.get("Question", "")))
            topic_ids = tuple(str(value) for value in row.get("topic_entity_ids", []))
            source_format = "official"
        answers, aliases = _alias_answers(answer_values)
        raw_id = row.get("id", row.get("_id", row.get("QuestionId", index)))
        return BenchmarkExample(
            example_id=str(raw_id),
            dataset=dataset,
            split=str(row.get("split", split)),
            question=question,
            topic_entity_ids=topic_ids,
            gold_answers=answers,
            answer_aliases=aliases,
            metadata={"source_format": source_format, "raw_index": index},
        )

    for index, example in enumerate(
        ordered_parallel_map(convert, enumerate(rows), workers=workers)
    ):
        examples.append(example)
        progress.update(index + 1)
    progress.finish(len(rows), examples=len(examples))
    return examples


def _ssp_rows(
    payload: Any,
    split: str,
    workers: int,
    include_datasets: frozenset[OpenQADataset],
) -> list[BenchmarkExample]:
    if not isinstance(payload, list):
        raise ValueError("SSP input must contain JSONL records")
    selected = [
        (index, row)
        for index, row in enumerate(payload)
        if _SSP_SOURCE_DATASETS.get(str(row.get("data_source"))) in include_datasets
    ]
    examples: list[BenchmarkExample] = []
    progress = ProgressLogger(f"data.prepare.ssp.parse.{split}", total=len(selected))
    progress.start(workers=workers)

    def convert(item: tuple[int, dict[str, Any]]) -> BenchmarkExample:
        raw_index, row = item
        data_source = str(row.get("data_source"))
        dataset = _SSP_SOURCE_DATASETS.get(data_source)
        if dataset is None:
            raise ValueError(f"unsupported SSP data_source: {data_source}")
        raw_extra = row.get("extra_info")
        extra: dict[str, Any] = raw_extra if isinstance(raw_extra, dict) else {}
        raw_reward = row.get("reward_model")
        reward: dict[str, Any] = raw_reward if isinstance(raw_reward, dict) else {}
        raw_ground_truth = reward.get("ground_truth")
        ground_truth: dict[str, Any] = (
            raw_ground_truth if isinstance(raw_ground_truth, dict) else {}
        )
        answers, aliases = _alias_answers(ground_truth.get("target", []))
        source_index = extra.get("index", raw_index)
        return BenchmarkExample(
            example_id=f"ssp:{data_source}:{source_index}",
            dataset=dataset,
            split=str(extra.get("split", split if split != "unknown" else "test")),
            question=str(extra.get("question", "")),
            topic_entity_ids=(),
            gold_answers=answers,
            answer_aliases=aliases,
            metadata={
                "source_format": "ssp",
                "data_source": data_source,
                "raw_index": raw_index,
                "source_index": source_index,
            },
        )

    for index, example in enumerate(ordered_parallel_map(convert, selected, workers=workers)):
        examples.append(example)
        progress.update(index + 1)
    progress.finish(len(selected), examples=len(examples))
    return examples


def _split_from_name(path: Path) -> str:
    lowered = path.name.casefold()
    for split in ("train", "dev", "val", "test"):
        if split in lowered:
            return "dev" if split == "val" else split
    return "unknown"


def prepare_benchmark(
    dataset: str,
    raw_dir: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    include_datasets: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    validate_workers(workers)
    adapters = {"webqsp": _webqsp_rows, "cwq": _cwq_rows, "grailqa": _grailqa_rows}
    openqa_datasets = frozenset((*COEVOKG_SSP_DATASETS, "musique"))
    if dataset not in adapters and dataset not in openqa_datasets and dataset != "ssp":
        raise ValueError(f"unsupported benchmark: {dataset}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if dataset == "ssp":
        selected_datasets = frozenset(
            cast(OpenQADataset, value) for value in (include_datasets or COEVOKG_SSP_DATASETS)
        )
        unknown = sorted(selected_datasets - openqa_datasets)
        if unknown:
            raise ValueError(f"unsupported SSP datasets: {', '.join(unknown)}")
    else:
        selected_datasets = frozenset()
    examples: list[BenchmarkExample] = []
    sources: list[dict[str, Any]] = []
    for path, payload in _load_json_files(raw_dir, workers=workers):
        split = _split_from_name(path)
        if dataset == "ssp":
            parsed = _ssp_rows(payload, split, workers, selected_datasets)
        elif dataset in openqa_datasets:
            parsed = _openqa_rows(dataset, payload, split, workers)
        else:
            parsed = adapters[dataset](payload, split, workers)
        if limit is not None:
            parsed = parsed[: max(0, limit - len(examples))]
        examples.extend(parsed)
        sources.append({"path": path.name, "sha256": file_hash(path), "rows": len(parsed)})
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no JSON/JSONL benchmark examples found under {raw_dir}")
    by_split: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_split.setdefault(example.split, []).append(example.model_dump(mode="json"))
    for split, rows in by_split.items():
        write_records(output_dir / split / "examples.parquet", rows)
    heldout = sorted(
        {
            entity_id
            for example in examples
            if example.split in {"dev", "test"}
            for entity_id in example.topic_entity_ids
        }
    )
    write_json(output_dir / "heldout_topic_entities.json", heldout)
    summary = {
        "dataset": dataset,
        "examples": len(examples),
        "splits": {split: len(rows) for split, rows in by_split.items()},
        "datasets": {
            value: sum(example.dataset == value for example in examples)
            for value in sorted({example.dataset for example in examples})
        },
        "heldout_topic_entities": len(heldout),
        "sources": sources,
        "workers": workers,
        "limit": limit,
    }
    if dataset == "ssp":
        exact_release_match = len(sources) == 1 and sources[0]["sha256"] == SSP_TEST_SHA256
        actual_counts = cast(dict[str, int], summary["datasets"])
        expected_counts = {value: SSP_EXPECTED_COUNTS[value] for value in sorted(selected_datasets)}
        if exact_release_match and limit is None and actual_counts != expected_counts:
            raise ValueError(
                f"SSP fixed revision bucket mismatch: expected {expected_counts}, "
                f"received {actual_counts}"
            )
        summary["ssp_protocol"] = {
            "revision": SSP_REVISION,
            "test_sha256": SSP_TEST_SHA256,
            "exact_release_match": exact_release_match,
            "expected_counts": expected_counts,
            "coevokg_test_parity": bool(
                exact_release_match and limit is None and actual_counts == expected_counts
            ),
        }
    write_json(output_dir / "metrics.json", summary)
    write_manifest(
        output_dir,
        {
            "command": "data prepare",
            "dataset": dataset,
            "workers": workers,
            "include_datasets": sorted(selected_datasets),
            "limit": limit,
        },
        ["*/examples.parquet", "heldout_topic_entities.json", "metrics.json"],
    )
    return summary
