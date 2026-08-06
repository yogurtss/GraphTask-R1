from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from graphtask_r1.schema import Answer, AnswerSet, BenchmarkExample
from graphtask_r1.utils import file_hash, write_json, write_manifest, write_records

_MID = re.compile(r"(?:ns:|/ns/|rdf\.freebase\.com/ns/)([mg]\.[A-Za-z0-9_]+)")


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


def _load_json_files(raw_dir: Path) -> list[tuple[Path, Any]]:
    files = sorted([*raw_dir.glob("*.json"), *raw_dir.glob("*.jsonl")])
    loaded: list[tuple[Path, Any]] = []
    for path in files:
        if path.suffix == ".jsonl":
            loaded.append(
                (path, [json.loads(line) for line in path.read_text().splitlines() if line])
            )
        else:
            loaded.append((path, json.loads(path.read_text())))
    return loaded


def _webqsp_rows(payload: Any, split: str) -> list[BenchmarkExample]:
    rows = payload.get("Questions", payload) if isinstance(payload, dict) else payload
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(rows):
        parses = row.get("Parses", [])
        parse = next(
            (value for value in parses if value.get("Sparql")), parses[0] if parses else {}
        )
        sparql = parse.get("Sparql")
        all_answers = [answer for value in parses for answer in value.get("Answers", [])]
        all_topics = [value.get("TopicEntityMid") for value in parses]
        examples.append(
            BenchmarkExample(
                example_id=str(row.get("QuestionId", row.get("id", index))),
                dataset="webqsp",
                split=split,
                question=str(row.get("RawQuestion", row.get("ProcessedQuestion", ""))),
                topic_entity_ids=_topics(sparql, all_topics),
                gold_answers=_answers(all_answers or row.get("Answers", [])),
                sparql=sparql,
                metadata={"parse_count": len(parses)},
            )
        )
    return examples


def _cwq_rows(payload: Any, split: str) -> list[BenchmarkExample]:
    rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(rows):
        sparql = row.get("sparql") or row.get("SPARQL")
        examples.append(
            BenchmarkExample(
                example_id=str(row.get("ID", row.get("id", index))),
                dataset="cwq",
                split=str(row.get("split", split)),
                question=str(row.get("question", row.get("machine_question", ""))),
                topic_entity_ids=_topics(sparql, row.get("topic_entities")),
                gold_answers=_answers(row.get("answers", row.get("answer", []))),
                logical_form=row.get("s_expression") or row.get("logical_form"),
                sparql=sparql,
            )
        )
    return examples


def _grailqa_rows(payload: Any, split: str) -> list[BenchmarkExample]:
    rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(rows):
        graph_query = row.get("graph_query", {})
        nodes = graph_query.get("nodes", []) if isinstance(graph_query, dict) else []
        explicit = [
            node.get("id")
            for node in nodes
            if node.get("node_type") == "entity" and not node.get("question_node", False)
        ]
        examples.append(
            BenchmarkExample(
                example_id=str(row.get("qid", row.get("id", index))),
                dataset="grailqa",
                split=str(row.get("split", split)),
                question=str(row.get("question", "")),
                topic_entity_ids=_topics(None, explicit),
                gold_answers=_answers(row.get("answer", row.get("answers", []))),
                logical_form=row.get("s_expression") or row.get("logical_form"),
                metadata={"function": row.get("function")},
            )
        )
    return examples


def _split_from_name(path: Path) -> str:
    lowered = path.name.casefold()
    for split in ("train", "dev", "val", "test"):
        if split in lowered:
            return "dev" if split == "val" else split
    return "unknown"


def prepare_benchmark(dataset: str, raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    adapters = {"webqsp": _webqsp_rows, "cwq": _cwq_rows, "grailqa": _grailqa_rows}
    if dataset not in adapters:
        raise ValueError(f"unsupported benchmark: {dataset}")
    examples: list[BenchmarkExample] = []
    sources: list[dict[str, Any]] = []
    for path, payload in _load_json_files(raw_dir):
        split = _split_from_name(path)
        parsed = adapters[dataset](payload, split)
        examples.extend(parsed)
        sources.append({"path": path.name, "sha256": file_hash(path), "rows": len(parsed)})
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
        "heldout_topic_entities": len(heldout),
        "sources": sources,
    }
    write_json(output_dir / "metrics.json", summary)
    write_manifest(
        output_dir,
        {"command": "data prepare", "dataset": dataset},
        ["*/examples.parquet", "heldout_topic_entities.json", "metrics.json"],
    )
    return summary
