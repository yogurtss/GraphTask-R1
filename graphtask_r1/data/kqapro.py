from __future__ import annotations

import json
import sqlite3
from collections import defaultdict, deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

from graphtask_r1.dsl import canonical_signature, compile_sparql, operator_tags, program_cost
from graphtask_r1.generation import compile_trace
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    LiteralValue,
    Program,
    TaskCertificate,
    TaskProvenance,
    Union,
    VerificationSummary,
)
from graphtask_r1.utils import (
    ProgressLogger,
    file_hash,
    stable_hash,
    write_json,
    write_manifest,
    write_records,
)
from graphtask_r1.verification import verify_task

CONVERTER_VERSION = "kqapro-core-v1"
UNSUPPORTED_KOPL = {
    "QFilterStr",
    "QFilterNum",
    "QFilterYear",
    "QFilterDate",
    "SelectBetween",
    "SelectAmong",
    "QueryAttr",
    "QueryAttrUnderCondition",
    "VerifyStr",
    "VerifyNum",
    "VerifyYear",
    "VerifyDate",
    "QueryRelation",
    "QueryAttrQualifier",
    "QueryRelationQualifier",
}


class KoPLConversionError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code


def _ancestors(concepts: dict[str, dict[str, Any]], direct: list[str]) -> set[str]:
    found: set[str] = set()
    queue = deque(direct)
    while queue:
        concept_id = queue.popleft()
        if concept_id in found:
            continue
        found.add(concept_id)
        queue.extend(concepts.get(concept_id, {}).get("instanceOf", []))
    return found


def build_kqapro_database(kb_path: Path, output_path: Path) -> dict[str, Any]:
    """Convert KQA Pro kb.json into an indexed, immutable-after-build SQLite snapshot."""
    payload = json.loads(kb_path.read_text())
    concepts: dict[str, dict[str, Any]] = payload["concepts"]
    entities: dict[str, dict[str, Any]] = payload["entities"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".building.sqlite")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE entities(
            entity_id TEXT PRIMARY KEY, label TEXT NOT NULL, aliases_json TEXT NOT NULL
        );
        CREATE TABLE entity_types(
            entity_id TEXT NOT NULL, type_id TEXT NOT NULL, PRIMARY KEY(entity_id, type_id)
        );
        CREATE TABLE triples(subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
            PRIMARY KEY(subject, relation, object));
        CREATE TABLE attributes(entity_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            datatype TEXT NOT NULL, unit TEXT, PRIMARY KEY(entity_id, key, value, datatype, unit));
        CREATE TABLE relation_labels(relation_id TEXT PRIMARY KEY, label TEXT NOT NULL);
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX idx_triples_subject_relation ON triples(subject, relation);
        CREATE INDEX idx_triples_object_relation ON triples(object, relation);
        CREATE INDEX idx_types_type_entity ON entity_types(type_id, entity_id);
        CREATE INDEX idx_attributes_entity_key ON attributes(entity_id, key);
        """
    )
    progress = ProgressLogger(
        "data.prepare.kqapro.build_graph", total=len(concepts) + len(entities)
    )
    progress.start(concepts=len(concepts), entities=len(entities))
    completed = 0
    relation_ids: set[str] = set()
    for concept_id, info in concepts.items():
        connection.execute(
            "INSERT INTO entities VALUES (?, ?, ?)",
            (concept_id, " ".join(str(info["name"]).split()), "[]"),
        )
        for type_id in _ancestors(concepts, list(info.get("instanceOf", []))):
            connection.execute(
                "INSERT OR IGNORE INTO entity_types VALUES (?, ?)", (concept_id, type_id)
            )
        completed += 1
        progress.update(completed, stage="concepts")
    for entity_id, info in entities.items():
        aliases = info.get("aliases", [])
        connection.execute(
            "INSERT INTO entities VALUES (?, ?, ?)",
            (
                entity_id,
                " ".join(str(info["name"]).split()),
                json.dumps(aliases, ensure_ascii=False),
            ),
        )
        for type_id in _ancestors(concepts, list(info.get("instanceOf", []))):
            connection.execute(
                "INSERT OR IGNORE INTO entity_types VALUES (?, ?)", (entity_id, type_id)
            )
        for relation in info.get("relations", []):
            predicate = str(relation["predicate"])
            relation_ids.add(predicate)
            if relation.get("direction") == "backward":
                subject, object_id = str(relation["object"]), entity_id
            else:
                subject, object_id = entity_id, str(relation["object"])
            connection.execute(
                "INSERT OR IGNORE INTO triples VALUES (?, ?, ?)",
                (subject, predicate, object_id),
            )
        for attribute in info.get("attributes", []):
            key = str(attribute["key"])
            relation_ids.add(key)
            raw_value = attribute["value"]
            value = str(raw_value["value"])
            datatype = str(raw_value["type"])
            unit = raw_value.get("unit")
            connection.execute(
                "INSERT OR IGNORE INTO attributes VALUES (?, ?, ?, ?, ?)",
                (entity_id, key, value, datatype, unit),
            )
            connection.execute(
                "INSERT OR IGNORE INTO triples VALUES (?, ?, ?)", (entity_id, key, value)
            )
        completed += 1
        progress.update(completed, stage="entities", relations=len(relation_ids))
    connection.executemany(
        "INSERT INTO relation_labels VALUES (?, ?)", ((value, value) for value in relation_ids)
    )
    metadata = {
        "snapshot_id": "kqapro-v1",
        "converter_version": CONVERTER_VERSION,
        "source_hash": file_hash(kb_path),
        "entities": len(entities),
        "concepts": len(concepts),
    }
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        ((key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()),
    )
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    temporary.replace(output_path)
    progress.finish(completed, relations=len(relation_ids), output=str(output_path))
    return metadata


class KoPLMapper:
    def __init__(self, backend: SQLiteGraphBackend) -> None:
        rows = backend.connection.execute("SELECT entity_id, label FROM entities").fetchall()
        self.name_to_ids: dict[str, list[str]] = defaultdict(list)
        for entity_id, label in rows:
            self.name_to_ids[str(label)].append(str(entity_id))
        type_rows = backend.connection.execute(
            "SELECT e.entity_id, e.label FROM entities e "
            "WHERE e.entity_id IN (SELECT DISTINCT type_id FROM entity_types)"
        ).fetchall()
        self.type_name_to_ids: dict[str, list[str]] = defaultdict(list)
        for type_id, label in type_rows:
            self.type_name_to_ids[str(label)].append(str(type_id))

    @staticmethod
    def _branch(values: list[str]) -> Program:
        if not values:
            raise KoPLConversionError("UNKNOWN_ENTITY", "no entity matches the KoPL name")
        programs = tuple(Entity(entity_id=value) for value in sorted(set(values)))
        return programs[0] if len(programs) == 1 else Union(inputs=programs)

    def convert(self, steps: list[dict[str, Any]]) -> Program:
        outputs: list[Program] = []
        for index, step in enumerate(steps):
            function = str(step["function"])
            inputs = [str(value) for value in step.get("inputs", [])]
            dependencies = [int(value) for value in step.get("dependencies", [])]
            if function in UNSUPPORTED_KOPL:
                raise KoPLConversionError("UNSUPPORTED_KOPL_OPERATOR", function)
            try:
                args = [outputs[value] for value in dependencies]
            except IndexError as exc:
                raise KoPLConversionError(
                    "INVALID_KOPL_DEPENDENCY", f"step {index}: {dependencies}"
                ) from exc
            if function == "FindAll":
                output: Program = AllEntities(max_results=1_000_000)
            elif function == "Find":
                output = self._branch(self.name_to_ids.get(inputs[0], []))
            elif function == "Relate":
                output = Hop(
                    input=args[0],
                    relation=inputs[0],
                    direction="out" if inputs[1] == "forward" else "in",
                )
            elif function == "And":
                output = Intersect(inputs=(args[0], args[1]))
            elif function == "Or":
                output = Union(inputs=(args[0], args[1]))
            elif function == "FilterConcept":
                type_ids = self.type_name_to_ids.get(inputs[0], [])
                if not type_ids:
                    raise KoPLConversionError("UNKNOWN_CONCEPT", inputs[0])
                filters = tuple(FilterType(input=args[0], type_id=value) for value in type_ids)
                output = filters[0] if len(filters) == 1 else Union(inputs=filters)
            elif function in {"FilterStr", "FilterNum", "FilterYear", "FilterDate"}:
                comparator = "eq" if function == "FilterStr" else _comparator(inputs[2])
                datatype = cast(
                    Literal["string", "quantity", "year", "date", "number"],
                    {
                        "FilterStr": "string",
                        "FilterNum": "quantity",
                        "FilterYear": "year",
                        "FilterDate": "date",
                    }[function],
                )
                raw_value = inputs[1]
                unit: str | None = None
                value: str | int | float = raw_value
                if datatype == "quantity":
                    parts = raw_value.split(maxsplit=1)
                    try:
                        value = float(parts[0])
                    except ValueError:
                        value = parts[0]
                    unit = parts[1] if len(parts) == 2 else None
                elif datatype == "year":
                    with suppress(ValueError):
                        value = int(raw_value)
                output = FilterLiteral(
                    input=args[0],
                    relation=inputs[0],
                    comparator=comparator,
                    value=LiteralValue(value=value, datatype=datatype, unit=unit),
                )
            elif function == "Count":
                output = Count(input=args[0])
            elif function == "What":
                output = args[0]
            else:
                raise KoPLConversionError("UNSUPPORTED_KOPL_OPERATOR", function)
            outputs.append(output)
        if not outputs:
            raise KoPLConversionError("EMPTY_KOPL_PROGRAM", "program has no steps")
        return outputs[-1]


def _comparator(
    value: str,
) -> Literal["eq", "ne", "lt", "le", "gt", "ge", "contains"]:
    mapping = {"=": "eq", "!=": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le"}
    if value not in mapping:
        raise KoPLConversionError("UNSUPPORTED_COMPARATOR", value)
    return cast(Literal["eq", "ne", "lt", "le", "gt", "ge", "contains"], mapping[value])


def _topic_ids(program: Program) -> tuple[str, ...]:
    if isinstance(program, Entity):
        return (program.entity_id,)
    if isinstance(program, AllEntities):
        return ()
    if isinstance(program, Intersect | Union):
        return tuple(sorted({value for branch in program.inputs for value in _topic_ids(branch)}))
    if isinstance(program, Hop | FilterType | FilterLiteral | Count):
        return _topic_ids(program.input)
    raise TypeError(type(program).__name__)


def _answer_matches_source(answer: str, derived: AnswerSet, backend: SQLiteGraphBackend) -> bool:
    candidates = {str(value) for value in derived.values()}
    candidates.update(backend.entity_info(value).label for value in derived.entity_ids())
    return answer.strip() in candidates


def prepare_kqapro(
    raw_dir: Path,
    output_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    kb_path = raw_dir / "kb.json"
    database_path = output_dir / "graph.sqlite"
    build_metadata = build_kqapro_database(kb_path, database_path)
    backend = SQLiteGraphBackend(database_path)
    mapper = KoPLMapper(backend)
    total_tasks = 0
    total_rejections = 0
    split_metrics: dict[str, dict[str, int]] = {}
    for split in splits:
        source_path = raw_dir / f"{split}.json"
        rows: list[dict[str, Any]] = json.loads(source_path.read_text())
        if limit is not None:
            rows = rows[:limit]
        tasks: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        progress = ProgressLogger(f"data.prepare.kqapro.split.{split}", total=len(rows))
        progress.start(source=str(source_path))
        for index, row in enumerate(rows):
            try:
                program = mapper.convert(row["program"])
                answers = backend.execute_program(program)
                if not answers.answers:
                    raise KoPLConversionError(
                        "EMPTY_ANSWER", "converted program returned no answer"
                    )
                if "answer" in row and not _answer_matches_source(
                    str(row["answer"]), answers, backend
                ):
                    raise KoPLConversionError(
                        "SOURCE_ANSWER_MISMATCH",
                        f"source={row['answer']!r}, derived={answers.values()!r}",
                    )
                verification = verify_task(str(row["question"]), program, backend)
                if not verification.passed:
                    raise KoPLConversionError(
                        "VERIFICATION_REJECTED", ",".join(verification.rejection_reasons)
                    )
                signature = canonical_signature(program)
                source_id = str(row.get("id", index))
                task_id = f"gt_kqapro_{split}_{stable_hash([source_id, signature])[:16]}"
                witnesses = backend.extract_witness(program, answers)
                task = TaskCertificate(
                    task_id=task_id,
                    source="kqapro",
                    source_id=source_id,
                    split=split,
                    graph_snapshot="kqapro-v1",
                    question=str(row["question"]),
                    topic_entities=tuple(
                        backend.entity_info(value) for value in _topic_ids(program)
                    ),
                    program=program,
                    sparql=compile_sparql(program),
                    gold_answers=answers,
                    witness_facts=tuple(
                        sorted(
                            {fact for witness in witnesses for fact in witness.facts},
                            key=lambda fact: fact.sort_key(),
                        )
                    ),
                    program_signature=signature,
                    program_cost=program_cost(program),
                    operator_tags=operator_tags(program),
                    verification=VerificationSummary(
                        executable=True,
                        semantic_equivalent=True,
                        necessity_mean=verification.necessity_mean,
                        necessity_min=verification.necessity_min,
                        shortcut_found=verification.shortcut_found,
                        answer_leak=verification.answer_leak,
                    ),
                    source_program={"language": "kopl", "steps": row["program"]},
                    provenance=TaskProvenance(
                        dataset="kqapro",
                        raw_file=source_path.name,
                        raw_index=index,
                        converter_version=CONVERTER_VERSION,
                        source_hash=file_hash(source_path),
                    ),
                    generation={"seed": seed, "graph_snapshot": "kqapro-v1"},
                )
                trace = compile_trace(task_id, task.question, program, backend, seed=seed + index)
                if trace.final_answers != answers:
                    raise KoPLConversionError("TRACE_REPLAY_MISMATCH", task_id)
                tasks.append(task.model_dump(mode="json"))
                traces.append(trace.model_dump(mode="json"))
            except (KoPLConversionError, TypeError, ValueError, KeyError, RuntimeError) as exc:
                reason = (
                    exc.reason_code if isinstance(exc, KoPLConversionError) else "CONVERSION_ERROR"
                )
                rejections.append(
                    {
                        "index": index,
                        "source_id": str(row.get("id", index)),
                        "reason_code": reason,
                        "detail": str(exc),
                    }
                )
            progress.update(
                index + 1,
                accepted=len(tasks),
                rejected=len(rejections),
            )
        split_dir = output_dir / split
        write_records(split_dir / "tasks.parquet", tasks)
        write_records(split_dir / "traces.parquet", traces)
        write_records(split_dir / "rejections.parquet", rejections)
        progress.finish(
            len(rows),
            accepted=len(tasks),
            rejected=len(rejections),
            output=str(split_dir),
        )
        split_metrics[split] = {
            "input": len(rows),
            "accepted": len(tasks),
            "rejected": len(rejections),
        }
        write_json(split_dir / "metrics.json", split_metrics[split])
        total_tasks += len(tasks)
        total_rejections += len(rejections)
    backend.close()
    summary = {
        "dataset": "kqapro",
        "snapshot": "kqapro-v1",
        "database": str(database_path),
        "build": build_metadata,
        "splits": split_metrics,
        "accepted": total_tasks,
        "rejected": total_rejections,
    }
    write_json(output_dir / "metrics.json", summary)
    write_manifest(
        output_dir,
        {"command": "data prepare", "dataset": "kqapro", "seed": seed},
        ["graph.sqlite", "*/tasks.parquet", "*/traces.parquet", "*/rejections.parquet"],
    )
    return summary
