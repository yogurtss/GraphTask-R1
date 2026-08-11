from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from graphtask_r1.utils import ProgressLogger, write_json, write_manifest, write_records

KILT_CONVERTER_VERSION = "kilt-hyperlink-v2"
KILT_SNAPSHOT_ID = "kilt-2019-08-01-v1"
KILT_EXPECTED_PAGES = 5_903_530
KILT_SOURCE_URL = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"
WIKIPEDIA_LINK_RELATION = "wikipedia_link"


def _metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        finally:
            connection.close()
        return {str(key): json.loads(value) for key, value in rows}
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return None


def _schema(connection: sqlite3.Connection, *, with_text_index: bool) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA locking_mode=EXCLUSIVE;
        CREATE TABLE entities(
            entity_id TEXT PRIMARY KEY, label TEXT NOT NULL, aliases_json TEXT NOT NULL
        );
        CREATE TABLE entity_types(
            entity_id TEXT NOT NULL, type_id TEXT NOT NULL, PRIMARY KEY(entity_id, type_id)
        );
        CREATE TABLE triples(
            subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
            PRIMARY KEY(subject, relation, object)
        );
        CREATE TABLE attributes(
            entity_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            datatype TEXT NOT NULL, unit TEXT, PRIMARY KEY(entity_id, key, value, datatype, unit)
        );
        CREATE TABLE relation_labels(relation_id TEXT PRIMARY KEY, label TEXT NOT NULL);
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE rejections(
            raw_index INTEGER NOT NULL, reason_code TEXT NOT NULL, detail TEXT NOT NULL,
            count INTEGER NOT NULL
        );
        CREATE INDEX idx_triples_subject_relation ON triples(subject, relation);
        CREATE INDEX idx_triples_object_relation ON triples(object, relation);
        CREATE INDEX idx_types_type_entity ON entity_types(type_id, entity_id);
        CREATE INDEX idx_attributes_entity_key ON attributes(entity_id, key);
        """
    )
    if with_text_index:
        connection.execute(
            "CREATE VIRTUAL TABLE passage_fts USING fts5("
            "page_id UNINDEXED, paragraph_id UNINDEXED, title, text, tokenize='unicode61')"
        )


def build_kilt_database(
    source_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    with_text_index: bool = True,
    snapshot_id: str = KILT_SNAPSHOT_ID,
) -> dict[str, Any]:
    """Stream the KILT JSONL knowledge source into the existing SQLite graph contract."""

    if limit is not None and limit < 1:
        raise ValueError("KILT build limit must be at least 1")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".building.sqlite")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    _schema(connection, with_text_index=with_text_index)
    connection.execute(
        "INSERT INTO relation_labels VALUES (?, ?)",
        (WIKIPEDIA_LINK_RELATION, "Wikipedia link"),
    )

    source_stat = source_path.stat()
    progress = ProgressLogger(
        "data.prepare.kilt.build_graph",
        total=limit if limit is not None else KILT_EXPECTED_PAGES,
    )
    progress.start(source=str(source_path), text_index=with_text_index)
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    complete = True
    try:
        with source_path.open("rb") as source:
            for raw_index, raw_line in enumerate(source):
                if limit is not None and raw_index >= limit:
                    complete = False
                    break
                digest.update(raw_line)
                counts["records"] += 1
                try:
                    page = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    connection.execute(
                        "INSERT INTO rejections VALUES (?, ?, ?, ?)",
                        (raw_index, "INVALID_JSON", str(exc), 1),
                    )
                    counts["rejections"] += 1
                    continue
                page_id = page.get("wikipedia_id", page.get("_id"))
                title = str(page.get("wikipedia_title", "")).strip()
                if page_id is None or not title:
                    connection.execute(
                        "INSERT INTO rejections VALUES (?, ?, ?, ?)",
                        (
                            raw_index,
                            "MISSING_PAGE_ID_OR_TITLE",
                            f"page_id={page_id!r}",
                            1,
                        ),
                    )
                    counts["rejections"] += 1
                    continue
                entity_id = str(page_id)
                wikidata = page.get("wikidata_info")
                wikidata_id = (
                    str(wikidata.get("wikidata_id"))
                    if isinstance(wikidata, dict) and wikidata.get("wikidata_id")
                    else None
                )
                aliases = [wikidata_id] if wikidata_id else []
                connection.execute(
                    "INSERT OR REPLACE INTO entities VALUES (?, ?, ?)",
                    (entity_id, title, json.dumps(aliases, ensure_ascii=False)),
                )
                if wikidata_id:
                    connection.execute(
                        "INSERT OR IGNORE INTO attributes VALUES (?, ?, ?, ?, ?)",
                        (entity_id, "wikidata_id", wikidata_id, "string", None),
                    )
                categories = str(page.get("categories", ""))
                for category in (value.strip() for value in categories.split(",")):
                    if category:
                        connection.execute(
                            "INSERT OR IGNORE INTO entity_types VALUES (?, ?)",
                            (entity_id, f"category:{category}"),
                        )
                        counts["categories"] += 1
                paragraphs = page.get("text", [])
                if not isinstance(paragraphs, list):
                    paragraphs = []
                if with_text_index:
                    connection.executemany(
                        "INSERT INTO passage_fts(page_id, paragraph_id, title, text) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            (entity_id, paragraph_id, title, str(text))
                            for paragraph_id, text in enumerate(paragraphs)
                            if str(text).strip()
                        ),
                    )
                counts["passages"] += sum(bool(str(text).strip()) for text in paragraphs)
                anchors = page.get("anchors", [])
                if not isinstance(anchors, list):
                    anchors = []
                missing_anchor_targets = 0
                for anchor in anchors:
                    target_id = anchor.get("wikipedia_id") if isinstance(anchor, dict) else None
                    if target_id is None or not str(target_id).strip():
                        missing_anchor_targets += 1
                        counts["rejections"] += 1
                        continue
                    connection.execute(
                        "INSERT OR IGNORE INTO triples VALUES (?, ?, ?)",
                        (entity_id, WIKIPEDIA_LINK_RELATION, str(target_id)),
                    )
                    counts["anchors"] += 1
                if missing_anchor_targets:
                    connection.execute(
                        "INSERT INTO rejections VALUES (?, ?, ?, ?)",
                        (
                            raw_index,
                            "MISSING_ANCHOR_TARGET",
                            f"page_id={entity_id}",
                            missing_anchor_targets,
                        ),
                    )
                counts["pages"] += 1
                if counts["records"] % 10_000 == 0:
                    connection.commit()
                progress.update(
                    counts["records"],
                    pages=counts["pages"],
                    anchors=counts["anchors"],
                    rejections=counts["rejections"],
                )
        metadata = {
            "snapshot_id": snapshot_id,
            "converter_version": KILT_CONVERTER_VERSION,
            "source_url": KILT_SOURCE_URL,
            "source_path": source_path.name,
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_complete": complete,
            "source_sha256": digest.hexdigest() if complete else None,
            "source_prefix_sha256": digest.hexdigest(),
            "source_records_read": counts["records"],
            "limit": limit,
            "with_text_index": with_text_index,
            **dict(counts),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                for key, value in metadata.items()
            ),
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.close()
        temporary.replace(output_path)
    except BaseException:
        connection.close()
        raise
    progress.finish(
        counts["records"],
        pages=counts["pages"],
        anchors=counts["anchors"],
        rejections=counts["rejections"],
        output=str(output_path),
    )
    return metadata


def prepare_kilt(
    source_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    with_text_index: bool = True,
    rebuild_graph: bool = False,
) -> dict[str, Any]:
    database_path = output_dir / "graph.sqlite"
    source_stat = source_path.stat()
    existing = _metadata(database_path)
    reusable = bool(
        not rebuild_graph
        and existing
        and existing.get("converter_version") == KILT_CONVERTER_VERSION
        and existing.get("source_size_bytes") == source_stat.st_size
        and existing.get("source_mtime_ns") == source_stat.st_mtime_ns
        and existing.get("limit") == limit
        and existing.get("with_text_index") == with_text_index
    )
    metadata = (
        existing
        if reusable and existing is not None
        else build_kilt_database(
            source_path,
            database_path,
            limit=limit,
            with_text_index=with_text_index,
        )
    )
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        rejection_rows = [
            {
                "index": raw_index,
                "reason_code": reason_code,
                "detail": detail,
                "count": count,
            }
            for raw_index, reason_code, detail, count in connection.execute(
                "SELECT raw_index, reason_code, detail, count FROM rejections "
                "ORDER BY raw_index, reason_code, detail"
            )
        ]
    finally:
        connection.close()
    write_records(output_dir / "rejections.parquet", rejection_rows)
    summary = {**metadata, "build": {"reused": reusable}, "database": str(database_path)}
    write_json(output_dir / "metrics.json", summary)
    write_manifest(
        output_dir,
        {
            "command": "data prepare",
            "dataset": "kilt",
            "limit": limit,
            "with_text_index": with_text_index,
        },
        ["graph.sqlite", "rejections.parquet", "metrics.json"],
    )
    return summary
