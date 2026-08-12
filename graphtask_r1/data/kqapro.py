from __future__ import annotations

import heapq
import json
import sqlite3
import threading
from collections import Counter, defaultdict, deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

from graphtask_r1.dsl import canonical_signature, compile_sparql, operator_tags, program_cost
from graphtask_r1.generation import TraceCompilationError, compile_trace
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.graph.materialize import materialize_program
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Count,
    Entity,
    FilterLiteral,
    FilterQualifier,
    FilterType,
    Hop,
    Intersect,
    LiteralValue,
    Program,
    QueryAttribute,
    QueryAttributeQualifier,
    QueryAttributeUnderCondition,
    QueryRelation,
    QueryRelationQualifier,
    SelectAmong,
    SelectBetween,
    TaskCertificate,
    TaskProvenance,
    Triple,
    Union,
    VerificationSummary,
    Verify,
)
from graphtask_r1.utils import (
    ProgressLogger,
    RecordWriter,
    file_hash,
    iter_json_array,
    ordered_parallel_map,
    stable_hash,
    validate_workers,
    write_json,
    write_manifest,
)
from graphtask_r1.verification import verify_task

GRAPH_CONVERTER_VERSION = "kqapro-qualified-facts-v3"
TASK_CONVERTER_VERSION = "kqapro-complete-kopl-v5"
UNSUPPORTED_KOPL: set[str] = set()
KQAPRO_SAMPLER_VERSION = "kopl-stratified-v1"


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


def build_kqapro_database(
    kb_path: Path, output_path: Path, *, source_hash: str | None = None
) -> dict[str, Any]:
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
        CREATE TABLE facts(fact_id TEXT PRIMARY KEY, kind TEXT NOT NULL, subject TEXT NOT NULL,
            predicate TEXT NOT NULL, object_entity TEXT, value TEXT, datatype TEXT, unit TEXT);
        CREATE TABLE qualifiers(fact_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            datatype TEXT NOT NULL, unit TEXT);
        CREATE TABLE relation_labels(relation_id TEXT PRIMARY KEY, label TEXT NOT NULL);
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX idx_triples_subject_relation ON triples(subject, relation);
        CREATE INDEX idx_triples_object_relation ON triples(object, relation);
        CREATE INDEX idx_types_type_entity ON entity_types(type_id, entity_id);
        CREATE INDEX idx_attributes_entity_key ON attributes(entity_id, key);
        CREATE INDEX idx_facts_relation ON facts(kind, subject, predicate, object_entity);
        CREATE INDEX idx_facts_attribute ON facts(kind, subject, predicate, value);
        CREATE INDEX idx_qualifiers_fact_key ON qualifiers(fact_id, key);
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
        for relation_index, relation in enumerate(info.get("relations", [])):
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
            fact_id = f"relation:{entity_id}:{relation_index}"
            connection.execute(
                "INSERT INTO facts VALUES (?, 'relation', ?, ?, ?, NULL, NULL, NULL)",
                (fact_id, subject, predicate, object_id),
            )
            _insert_qualifiers(connection, fact_id, relation.get("qualifiers", {}), relation_ids)
        for attribute_index, attribute in enumerate(info.get("attributes", [])):
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
            fact_id = f"attribute:{entity_id}:{attribute_index}"
            connection.execute(
                "INSERT INTO facts VALUES (?, 'attribute', ?, ?, NULL, ?, ?, ?)",
                (fact_id, entity_id, key, value, datatype, unit),
            )
            _insert_qualifiers(connection, fact_id, attribute.get("qualifiers", {}), relation_ids)
        completed += 1
        progress.update(completed, stage="entities", relations=len(relation_ids))
    connection.executemany(
        "INSERT INTO relation_labels VALUES (?, ?)", ((value, value) for value in relation_ids)
    )
    fact_count = int(connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
    qualifier_count = int(connection.execute("SELECT COUNT(*) FROM qualifiers").fetchone()[0])
    metadata = {
        "snapshot_id": "kqapro-v1",
        "converter_version": GRAPH_CONVERTER_VERSION,
        "source_hash": source_hash if source_hash is not None else file_hash(kb_path),
        "entities": len(entities),
        "concepts": len(concepts),
        "facts": fact_count,
        "qualifiers": qualifier_count,
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


def _insert_qualifiers(
    connection: sqlite3.Connection,
    fact_id: str,
    qualifiers: dict[str, list[dict[str, Any]]],
    relation_ids: set[str],
) -> None:
    for key, raw_values in qualifiers.items():
        qualifier = str(key)
        relation_ids.add(qualifier)
        for raw_value in raw_values:
            connection.execute(
                "INSERT INTO qualifiers VALUES (?, ?, ?, ?, ?)",
                (
                    fact_id,
                    qualifier,
                    str(raw_value["value"]),
                    str(raw_value["type"]),
                    raw_value.get("unit"),
                ),
            )


def _existing_kqapro_database_metadata(database_path: Path) -> dict[str, Any] | None:
    if not database_path.exists():
        return None
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        finally:
            connection.close()
        return {str(key): json.loads(value) for key, value in rows}
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return None


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
            elif function in {"QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate"}:
                datatype = cast(
                    Literal["string", "quantity", "year", "date", "number"],
                    {
                        "QFilterStr": "string",
                        "QFilterNum": "quantity",
                        "QFilterYear": "year",
                        "QFilterDate": "date",
                    }[function],
                )
                output = FilterQualifier(
                    input=args[0],
                    qualifier=inputs[0],
                    comparator="eq" if function == "QFilterStr" else _comparator(inputs[2]),
                    value=_kopl_literal(inputs[1], datatype=datatype),
                )
            elif function == "Count":
                output = Count(input=args[0])
            elif function == "What":
                output = args[0]
            elif function == "QueryAttr":
                output = QueryAttribute(input=args[0], attribute=inputs[0])
            elif function == "QueryAttrUnderCondition":
                output = QueryAttributeUnderCondition(
                    input=args[0],
                    attribute=inputs[0],
                    qualifier=inputs[1],
                    qualifier_value=_kopl_literal(inputs[2]),
                )
            elif function == "QueryAttrQualifier":
                output = QueryAttributeQualifier(
                    input=args[0],
                    attribute=inputs[0],
                    attribute_value=_kopl_literal(inputs[1]),
                    qualifier=inputs[2],
                )
            elif function == "QueryRelation":
                output = QueryRelation(subject=args[0], object=args[1])
            elif function == "QueryRelationQualifier":
                output = QueryRelationQualifier(
                    subject=args[0],
                    object=args[1],
                    relation=inputs[0],
                    qualifier=inputs[1],
                )
            elif function in {"VerifyStr", "VerifyNum", "VerifyYear", "VerifyDate"}:
                datatype = cast(
                    Literal["string", "quantity", "year", "date", "number"],
                    {
                        "VerifyStr": "string",
                        "VerifyNum": "quantity",
                        "VerifyYear": "year",
                        "VerifyDate": "date",
                    }[function],
                )
                output = Verify(
                    input=args[0],
                    comparator="eq" if function == "VerifyStr" else _comparator(inputs[1]),
                    value=_kopl_literal(inputs[0], datatype=datatype),
                )
            elif function == "SelectBetween":
                output = SelectBetween(
                    left=args[0],
                    right=args[1],
                    attribute=inputs[0],
                    mode="max" if inputs[1] == "greater" else "min",
                )
            elif function == "SelectAmong":
                output = SelectAmong(
                    input=args[0],
                    attribute=inputs[0],
                    mode="max" if inputs[1] == "largest" else "min",
                )
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


def _kopl_literal(
    raw_value: str,
    *,
    datatype: str | None = None,
) -> LiteralValue:
    inferred = datatype
    value: str | int | float = raw_value
    unit: str | None = None
    if inferred is None:
        inferred = "string"
    elif inferred in {"quantity", "number"}:
        parts = raw_value.split(maxsplit=1)
        try:
            number = float(parts[0])
            value = int(number) if number.is_integer() else number
        except ValueError:
            value = parts[0]
        unit = parts[1] if len(parts) == 2 else None
    elif inferred == "year":
        with suppress(ValueError):
            value = int(raw_value)
    return LiteralValue(value=value, datatype=cast(Any, inferred), unit=unit)


def _topic_ids(program: Program) -> tuple[str, ...]:
    if isinstance(program, Entity):
        return (program.entity_id,)
    if isinstance(program, AllEntities):
        return ()
    if isinstance(program, Intersect | Union):
        return tuple(sorted({value for branch in program.inputs for value in _topic_ids(branch)}))
    if isinstance(
        program,
        Hop
        | FilterType
        | FilterLiteral
        | FilterQualifier
        | Count
        | QueryAttribute
        | QueryAttributeUnderCondition
        | QueryAttributeQualifier
        | Verify
        | SelectAmong,
    ):
        return _topic_ids(program.input)
    if isinstance(program, QueryRelation | QueryRelationQualifier):
        return tuple(sorted({*_topic_ids(program.subject), *_topic_ids(program.object)}))
    if isinstance(program, SelectBetween):
        return tuple(sorted({*_topic_ids(program.left), *_topic_ids(program.right)}))
    raise TypeError(type(program).__name__)


def _answer_matches_source(answer: str, derived: AnswerSet, backend: SQLiteGraphBackend) -> bool:
    candidates = {str(value) for value in derived.values()}
    candidates.update(info.label for info in backend.entity_infos(derived.entity_ids()))
    return answer.strip() in candidates


def _kopl_stratum(row: dict[str, Any]) -> str:
    raw_steps = row.get("program", [])
    steps = raw_steps if isinstance(raw_steps, list) else []
    operators = sorted(
        {
            str(step.get("function", "UNKNOWN"))
            for step in steps
            if isinstance(step, dict)
        }
    )
    terminal = (
        str(steps[-1].get("function", "UNKNOWN"))
        if steps and isinstance(steps[-1], dict)
        else "UNKNOWN"
    )
    length = len(steps)
    length_bucket = (
        "01-03" if length <= 3 else "04-06" if length <= 6 else "07-10" if length <= 10 else "11+"
    )
    return f"terminal={terminal}|steps={length_bucket}|ops={','.join(operators)}"


def _allocate_stratified_quotas(
    counts: Counter[str], sample_size: int
) -> dict[str, int]:
    population = sum(counts.values())
    target = min(sample_size, population)
    if target <= 0:
        return {key: 0 for key in counts}
    keys = sorted(counts)
    quotas = {key: 0 for key in keys}
    if target >= len(keys):
        quotas = {key: 1 for key in keys}
        remaining = target - len(keys)
    else:
        for key in sorted(keys, key=lambda value: (counts[value], value))[:target]:
            quotas[key] = 1
        return quotas
    capacities = {key: counts[key] - quotas[key] for key in keys}
    while remaining:
        capacity_total = sum(capacities.values())
        if capacity_total <= 0:
            break
        allocations: list[tuple[float, str, int]] = []
        assigned = 0
        for key in keys:
            ideal = remaining * capacities[key] / capacity_total
            whole = min(capacities[key], int(ideal))
            quotas[key] += whole
            capacities[key] -= whole
            assigned += whole
            allocations.append((ideal - whole, key, whole))
        remaining -= assigned
        if not remaining:
            break
        eligible = sorted(
            (value for value in allocations if capacities[value[1]] > 0),
            key=lambda value: (-value[0], value[1]),
        )
        if not eligible:
            break
        for _, key, _ in eligible[:remaining]:
            quotas[key] += 1
            capacities[key] -= 1
            remaining -= 1
    return quotas


def _stratified_kqapro_rows(
    source_path: Path,
    *,
    sample_size: int,
    seed: int,
    limit: int | None,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("train_sample_size must be at least 1")
    counts: Counter[str] = Counter()
    source_rows = 0
    source_operators: set[str] = set()
    for row in iter_json_array(source_path, limit=limit):
        counts[_kopl_stratum(row)] += 1
        source_rows += 1
        source_operators.update(
            str(step.get("function", "UNKNOWN"))
            for step in row.get("program", [])
            if isinstance(step, dict)
        )
    quotas = _allocate_stratified_quotas(counts, sample_size)
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(iter_json_array(source_path, limit=limit)):
        stratum = _kopl_stratum(row)
        quota = quotas[stratum]
        if not quota:
            continue
        priority = int(
            stable_hash(
                [
                    KQAPRO_SAMPLER_VERSION,
                    str(seed),
                    str(row.get("id", index)),
                    str(index),
                ]
            ),
            16,
        )
        candidate = (-priority, index, row)
        heap = heaps[stratum]
        if len(heap) < quota:
            heapq.heappush(heap, candidate)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, candidate)
    selected = sorted(
        ((index, row) for heap in heaps.values() for _, index, row in heap),
        key=lambda value: value[0],
    )
    selected_operators = sorted(
        {
            str(step.get("function", "UNKNOWN"))
            for _, row in selected
            for step in row.get("program", [])
            if isinstance(step, dict)
        }
    )
    metrics = {
        "sampler_version": KQAPRO_SAMPLER_VERSION,
        "seed": seed,
        "requested": sample_size,
        "source_rows": source_rows,
        "selected_rows": len(selected),
        "source_strata": len(counts),
        "selected_strata": sum(bool(value) for value in quotas.values()),
        "source_operators": sorted(source_operators),
        "selected_operators": selected_operators,
        "strata": {
            key: {"source": counts[key], "selected": quotas[key]} for key in sorted(counts)
        },
    }
    return selected, metrics


def _convert_kqapro_row_uncached(
    item: tuple[int, dict[str, Any]],
    *,
    mapper: KoPLMapper,
    backend: SQLiteGraphBackend,
    split: str,
    source_path: Path,
    source_hash: str,
    seed: int,
    max_trace_tool_calls: int,
    max_trace_query_results: int,
    max_witness_facts: int,
    trace_mode: Literal["none", "canonical"],
    verification_mode: Literal["source", "full"],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    index, row = item
    try:
        program = mapper.convert(row["program"])
        answers = backend.execute_program(program)
        if not answers.answers:
            raise KoPLConversionError("EMPTY_ANSWER", "converted program returned no answer")
        if "answer" in row and not _answer_matches_source(str(row["answer"]), answers, backend):
            raise KoPLConversionError(
                "SOURCE_ANSWER_MISMATCH",
                f"source={row['answer']!r}, derived={answers.values()!r}",
            )
        verification = (
            verify_task(str(row["question"]), program, backend)
            if verification_mode == "full"
            else None
        )
        if verification is not None and not verification.passed:
            raise KoPLConversionError(
                "VERIFICATION_REJECTED", ",".join(verification.rejection_reasons)
            )
        signature = canonical_signature(program)
        source_id = str(row.get("id", index))
        task_id = f"gt_kqapro_{split}_{stable_hash([source_id, signature])[:16]}"
        witness_facts: tuple[Triple, ...] = ()
        witness_complete = False
        witness_truncated = False
        witness_omitted = max_witness_facts == 0
        if max_witness_facts:
            witness_slice = materialize_program(
                backend,
                program,
                snapshot_id="kqapro-v1",
                max_nodes=max(100, max_witness_facts * 2 + 20),
                max_edges=max_witness_facts,
                include_neighborhood=False,
                include_metadata=False,
            )
            witness_facts = witness_slice.triples
            witness_complete = witness_slice.complete
            witness_truncated = witness_slice.truncated
        task = TaskCertificate(
            task_id=task_id,
            source="kqapro",
            source_id=source_id,
            split=split,
            graph_snapshot="kqapro-v1",
            question=str(row["question"]),
            topic_entities=backend.entity_infos(_topic_ids(program)),
            program=program,
            sparql=compile_sparql(program),
            gold_answers=answers,
            witness_facts=witness_facts,
            witness_complete=witness_complete,
            program_signature=signature,
            program_cost=program_cost(program),
            operator_tags=operator_tags(program),
            verification=VerificationSummary(
                executable=True,
                semantic_equivalent=True,
                necessity_mean=verification.necessity_mean if verification is not None else 0.0,
                necessity_min=verification.necessity_min if verification is not None else 0.0,
                shortcut_found=verification.shortcut_found if verification is not None else None,
                answer_leak=verification.answer_leak if verification is not None else False,
            ),
            source_program={"language": "kopl", "steps": row["program"]},
            provenance=TaskProvenance(
                dataset="kqapro",
                raw_file=source_path.name,
                raw_index=index,
                converter_version=TASK_CONVERTER_VERSION,
                source_hash=source_hash,
            ),
            generation={
                "seed": seed,
                "graph_snapshot": "kqapro-v1",
                "max_witness_facts": max_witness_facts,
                "witness_truncated": witness_truncated,
                "witness_omitted": witness_omitted,
                "trace_mode": trace_mode,
                "verification_mode": verification_mode,
            },
        )
        trace = None
        if trace_mode == "canonical":
            trace = compile_trace(
                task_id,
                task.question,
                program,
                backend,
                seed=seed + index,
                max_tool_calls=max_trace_tool_calls,
                max_query_results=max_trace_query_results,
            )
            if trace.final_answers != answers:
                raise KoPLConversionError("TRACE_REPLAY_MISMATCH", task_id)
        return (
            task.model_dump(mode="json"),
            trace.model_dump(mode="json") if trace is not None else None,
            None,
        )
    except (KoPLConversionError, TypeError, ValueError, KeyError, RuntimeError) as exc:
        reason = (
            exc.reason_code
            if isinstance(exc, KoPLConversionError | TraceCompilationError)
            else "CONVERSION_ERROR"
        )
        return (
            None,
            None,
            {
                "index": index,
                "source_id": str(row.get("id", index)),
                "reason_code": reason,
                "detail": str(exc),
            },
        )


def _convert_kqapro_row(
    item: tuple[int, dict[str, Any]],
    *,
    mapper: KoPLMapper,
    backend: SQLiteGraphBackend,
    split: str,
    source_path: Path,
    source_hash: str,
    seed: int,
    max_trace_tool_calls: int,
    max_trace_query_results: int,
    max_witness_facts: int,
    trace_mode: Literal["none", "canonical"],
    verification_mode: Literal["source", "full"],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    with backend.query_cache():
        return _convert_kqapro_row_uncached(
            item,
            mapper=mapper,
            backend=backend,
            split=split,
            source_path=source_path,
            source_hash=source_hash,
            seed=seed,
            max_trace_tool_calls=max_trace_tool_calls,
            max_trace_query_results=max_trace_query_results,
            max_witness_facts=max_witness_facts,
            trace_mode=trace_mode,
            verification_mode=verification_mode,
        )


class _KQAProWorker:
    def __init__(
        self,
        *,
        mapper: KoPLMapper,
        backend: SQLiteGraphBackend,
        database_path: Path,
        split: str,
        source_path: Path,
        source_hash: str,
        seed: int,
        max_trace_tool_calls: int,
        max_trace_query_results: int,
        max_witness_facts: int,
        trace_mode: Literal["none", "canonical"],
        verification_mode: Literal["source", "full"],
        parallel: bool,
    ) -> None:
        self.mapper = mapper
        self.backend = backend
        self.database_path = database_path
        self.split = split
        self.source_path = source_path
        self.source_hash = source_hash
        self.seed = seed
        self.max_trace_tool_calls = max_trace_tool_calls
        self.max_trace_query_results = max_trace_query_results
        self.max_witness_facts = max_witness_facts
        self.trace_mode = trace_mode
        self.verification_mode = verification_mode
        self.parallel = parallel
        self.local = threading.local()
        self.backends: list[SQLiteGraphBackend] = []
        self.backends_lock = threading.Lock()

    def __call__(
        self, item: tuple[int, dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        row_backend = self.backend
        if self.parallel:
            local_backend = cast(SQLiteGraphBackend | None, getattr(self.local, "backend", None))
            if local_backend is None:
                local_backend = SQLiteGraphBackend(self.database_path, allow_cross_thread=True)
                self.local.backend = local_backend
                with self.backends_lock:
                    self.backends.append(local_backend)
            row_backend = local_backend
        return _convert_kqapro_row(
            item,
            mapper=self.mapper,
            backend=row_backend,
            split=self.split,
            source_path=self.source_path,
            source_hash=self.source_hash,
            seed=self.seed,
            max_trace_tool_calls=self.max_trace_tool_calls,
            max_trace_query_results=self.max_trace_query_results,
            max_witness_facts=self.max_witness_facts,
            trace_mode=self.trace_mode,
            verification_mode=self.verification_mode,
        )

    def close(self) -> None:
        for backend in self.backends:
            backend.close()


def prepare_kqapro(
    raw_dir: Path,
    output_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    limit: int | None = None,
    seed: int = 42,
    workers: int = 1,
    rebuild_graph: bool = False,
    max_trace_tool_calls: int = 32,
    max_trace_query_results: int = 1_024,
    max_witness_facts: int = 0,
    train_sample_size: int | None = 20_000,
    trace_mode: Literal["none", "canonical"] = "none",
    verification_mode: Literal["source", "full"] = "source",
) -> dict[str, Any]:
    validate_workers(workers)
    if max_trace_tool_calls < 2:
        raise ValueError("max_trace_tool_calls must be at least 2")
    if not 1 <= max_trace_query_results <= 4_096:
        raise ValueError("max_trace_query_results must be between 1 and 4096")
    if not 0 <= max_witness_facts <= 50_000:
        raise ValueError("max_witness_facts must be between 0 and 50000")
    if train_sample_size is not None and train_sample_size < 1:
        raise ValueError("train_sample_size must be at least 1 or None")
    kb_path = raw_dir / "kb.json"
    database_path = output_dir / "graph.sqlite"
    kb_source_hash = file_hash(kb_path)
    if kb_source_hash is None:
        raise FileNotFoundError(kb_path)
    existing_metadata = _existing_kqapro_database_metadata(database_path)
    can_reuse_graph = (
        not rebuild_graph
        and existing_metadata is not None
        and existing_metadata.get("source_hash") == kb_source_hash
        and existing_metadata.get("converter_version") == GRAPH_CONVERTER_VERSION
        and existing_metadata.get("snapshot_id") == "kqapro-v1"
    )
    if can_reuse_graph:
        assert existing_metadata is not None
        build_metadata = {**existing_metadata, "reused": True}
    else:
        build_metadata = {
            **build_kqapro_database(kb_path, database_path, source_hash=kb_source_hash),
            "reused": False,
        }
    backend = SQLiteGraphBackend(database_path)
    mapper = KoPLMapper(backend)
    total_tasks = 0
    total_rejections = 0
    split_metrics: dict[str, dict[str, Any]] = {}
    for split in splits:
        source_path = raw_dir / f"{split}.json"
        source_hash = file_hash(source_path)
        if source_hash is None:
            raise FileNotFoundError(source_path)
        sampling_metrics: dict[str, Any] | None = None
        if split == "train" and train_sample_size is not None:
            selected_rows, sampling_metrics = _stratified_kqapro_rows(
                source_path,
                sample_size=train_sample_size,
                seed=seed,
                limit=limit,
            )
            indexed_rows: Any = iter(selected_rows)
            selected_total: int | None = len(selected_rows)
        else:
            indexed_rows = enumerate(iter_json_array(source_path, limit=limit))
            selected_total = limit
        accepted = 0
        rejected = 0
        trace_rows = 0
        rejection_reasons: Counter[str] = Counter()
        progress = ProgressLogger(f"data.prepare.kqapro.split.{split}", total=selected_total)
        progress.start(
            source=str(source_path),
            workers=workers,
            loading="stratified" if sampling_metrics is not None else "streaming",
            trace_mode=trace_mode,
            verification_mode=verification_mode,
        )
        process = _KQAProWorker(
            mapper=mapper,
            backend=backend,
            database_path=database_path,
            split=split,
            source_path=source_path,
            source_hash=source_hash,
            seed=seed,
            max_trace_tool_calls=max_trace_tool_calls,
            max_trace_query_results=max_trace_query_results,
            max_witness_facts=max_witness_facts,
            trace_mode=trace_mode,
            verification_mode=verification_mode,
            parallel=workers > 1,
        )

        split_dir = output_dir / split
        input_count = 0
        try:
            with (
                RecordWriter(split_dir / "tasks.parquet") as task_writer,
                RecordWriter(split_dir / "traces.parquet") as trace_writer,
                RecordWriter(split_dir / "rejections.parquet") as rejection_writer,
            ):
                converted = ordered_parallel_map(process, indexed_rows, workers=workers)
                for index, (task, trace, rejection) in enumerate(converted):
                    input_count = index + 1
                    if task is not None:
                        task_writer.write(task)
                        if trace is not None:
                            trace_writer.write(trace)
                            trace_rows += 1
                        accepted += 1
                    if rejection is not None:
                        rejection_writer.write(rejection)
                        rejected += 1
                        rejection_reasons[str(rejection["reason_code"])] += 1
                    progress.update(
                        index + 1,
                        accepted=accepted,
                        rejected=rejected,
                    )
        finally:
            process.close()
        progress.finish(
            input_count,
            accepted=accepted,
            rejected=rejected,
            output=str(split_dir),
        )
        split_metrics[split] = {
            "input": input_count,
            "accepted": accepted,
            "rejected": rejected,
            "traces": trace_rows,
            "acceptance_rate": accepted / input_count if input_count else 0.0,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        }
        if sampling_metrics is not None:
            write_json(split_dir / "sampling.json", sampling_metrics)
            split_metrics[split]["source_input"] = int(sampling_metrics["source_rows"])
        write_json(split_dir / "metrics.json", split_metrics[split])
        total_tasks += accepted
        total_rejections += rejected
    backend.close()
    summary = {
        "dataset": "kqapro",
        "snapshot": "kqapro-v1",
        "database": str(database_path),
        "build": build_metadata,
        "splits": split_metrics,
        "accepted": total_tasks,
        "rejected": total_rejections,
        "workers": workers,
        "max_trace_tool_calls": max_trace_tool_calls,
        "max_trace_query_results": max_trace_query_results,
        "max_witness_facts": max_witness_facts,
        "train_sample_size": train_sample_size,
        "trace_mode": trace_mode,
        "verification_mode": verification_mode,
    }
    write_json(output_dir / "metrics.json", summary)
    write_manifest(
        output_dir,
        {
            "command": "data prepare",
            "dataset": "kqapro",
            "seed": seed,
            "workers": workers,
            "max_trace_tool_calls": max_trace_tool_calls,
            "max_trace_query_results": max_trace_query_results,
            "max_witness_facts": max_witness_facts,
            "train_sample_size": train_sample_size,
            "trace_mode": trace_mode,
            "verification_mode": verification_mode,
        },
        [
            "graph.sqlite",
            "*/tasks.parquet",
            "*/traces.parquet",
            "*/rejections.parquet",
            "*/sampling.json",
        ],
    )
    return summary
