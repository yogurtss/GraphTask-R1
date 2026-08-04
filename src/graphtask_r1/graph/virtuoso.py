from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from graphtask_r1.graph.overlay import GraphOverlay
from graphtask_r1.schema import (
    Answer,
    AnswerSet,
    EntityInfo,
    Program,
    RelationInfo,
    Triple,
    Witness,
)


class VirtuosoBackend:
    """Minimal SPARQL 1.1 backend with timeout, retry, disk cache, and trace IDs."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 20.0,
        retries: int = 2,
        cache_path: Path = Path("data/cache/virtuoso.sqlite"),
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.retries = retries
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(cache_path)
        self.cache.execute(
            "CREATE TABLE IF NOT EXISTS queries (query_hash TEXT PRIMARY KEY, payload TEXT)"
        )
        self.cache.commit()

    def _query(self, sparql: str, *, trace_id: str | None = None) -> dict[str, Any]:
        query_hash = hashlib.sha256(sparql.encode()).hexdigest()
        cached = self.cache.execute(
            "SELECT payload FROM queries WHERE query_hash = ?", (query_hash,)
        ).fetchone()
        if cached is not None:
            return cast(dict[str, Any], json.loads(cached[0]))
        body = urllib.parse.urlencode(
            {"query": sparql, "format": "application/sparql-results+json"}
        )
        request = urllib.request.Request(
            self.endpoint,
            data=body.encode(),
            headers={
                "Accept": "application/sparql-results+json",
                "X-Trace-Id": trace_id or query_hash[:16],
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.cache.execute(
                    "INSERT OR REPLACE INTO queries(query_hash, payload) VALUES (?, ?)",
                    (query_hash, json.dumps(payload, sort_keys=True)),
                )
                self.cache.commit()
                return cast(dict[str, Any], payload)
            except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(0.25 * 2**attempt, 1.0))
        raise RuntimeError(
            f"SPARQL request failed after {self.retries + 1} attempts"
        ) from last_error

    def neighbors(
        self,
        entity_ids: Sequence[str],
        *,
        direction: str,
        relation_ids: Sequence[str] | None = None,
        limit: int = 100,
        trace_id: str | None = None,
    ) -> list[Triple]:
        from graphtask_r1.dsl.compiler import escape_iri

        if direction not in {"out", "in", "both"}:
            raise ValueError(f"invalid direction: {direction}")
        entities = " ".join(f"<{escape_iri(value)}>" for value in entity_ids)
        relation_filter = ""
        if relation_ids:
            relations = ", ".join(f"<{escape_iri(value)}>" for value in relation_ids)
            relation_filter = f"FILTER(?r IN ({relations}))"
        branches: list[str] = []
        if direction in {"out", "both"}:
            branches.append(f"{{ VALUES ?s {{ {entities} }} ?s ?r ?o . {relation_filter} }}")
        if direction in {"in", "both"}:
            branches.append(f"{{ VALUES ?o {{ {entities} }} ?s ?r ?o . {relation_filter} }}")
        query = (
            "SELECT DISTINCT ?s ?r ?o WHERE { "
            + " UNION ".join(branches)
            + f" }} ORDER BY ?s ?r ?o LIMIT {max(0, limit)}"
        )
        bindings = self._query(query, trace_id=trace_id)["results"]["bindings"]
        return [
            Triple(
                subject=str(row["s"]["value"]),
                relation=str(row["r"]["value"]),
                object=str(row["o"]["value"]),
            )
            for row in bindings
        ]

    def execute_program(self, program: Program) -> AnswerSet:
        from graphtask_r1.dsl.compiler import compile_sparql

        return self.execute_sparql(compile_sparql(program))

    def execute_sparql(self, sparql: str) -> AnswerSet:
        payload = self._query(sparql)
        variables = payload.get("head", {}).get("vars", [])
        bindings = payload.get("results", {}).get("bindings", [])
        if not variables:
            return AnswerSet()
        variable = str(variables[0])
        answers: list[Answer] = []
        for row in bindings:
            binding = row[variable]
            raw = str(binding["value"])
            if variable == "count":
                answers.append(Answer(value=int(raw), kind="count"))
            elif binding.get("type") == "uri":
                answers.append(Answer(value=raw, kind="entity"))
            else:
                answers.append(Answer(value=raw, kind="literal"))
        return AnswerSet(answers=tuple(answers))

    def entity_info(self, entity_id: str) -> EntityInfo:
        from graphtask_r1.dsl.compiler import escape_iri

        entity = escape_iri(entity_id)
        query = (
            "SELECT ?label ?type WHERE { "
            f"OPTIONAL {{ <{entity}> <http://www.w3.org/2000/01/rdf-schema#label> ?label . "
            "FILTER(lang(?label) = '' || langMatches(lang(?label), 'en')) }} "
            f"OPTIONAL {{ <{entity}> a ?type . }} }} ORDER BY ?label ?type LIMIT 100"
        )
        rows = self._query(query)["results"]["bindings"]
        labels = sorted({str(row["label"]["value"]) for row in rows if "label" in row})
        types = tuple(sorted({str(row["type"]["value"]) for row in rows if "type" in row}))
        return EntityInfo(
            entity_id=entity_id, label=labels[0] if labels else entity_id, type_ids=types
        )

    def relation_info(self, relation_id: str) -> RelationInfo:
        from graphtask_r1.dsl.compiler import escape_iri

        relation = escape_iri(relation_id)
        query = (
            "SELECT ?label WHERE { "
            f"<{relation}> <http://www.w3.org/2000/01/rdf-schema#label> ?label . "
            "FILTER(lang(?label) = '' || langMatches(lang(?label), 'en')) } LIMIT 1"
        )
        rows = self._query(query)["results"]["bindings"]
        label = str(rows[0]["label"]["value"]) if rows else relation_id
        return RelationInfo(relation_id=relation_id, label=label)

    def extract_witness(self, program: Program, answers: AnswerSet) -> list[Witness]:
        del program
        return [Witness(answer=str(answer.value), facts=()) for answer in answers.answers]

    def with_overlay(self, overlay: GraphOverlay) -> VirtuosoBackend:
        if overlay.added or overlay.removed:
            raise NotImplementedError(
                "materialize the bounded local witness subgraph before applying graph overlays"
            )
        return self
