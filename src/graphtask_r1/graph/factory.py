from __future__ import annotations

import os
from pathlib import Path

from graphtask_r1.graph.base import GraphBackend
from graphtask_r1.graph.memory import toy_graph
from graphtask_r1.graph.sqlite import SQLiteGraphBackend
from graphtask_r1.graph.virtuoso import VirtuosoBackend


def backend_from_snapshot(snapshot: str) -> GraphBackend:
    if snapshot == "toy-v1":
        return toy_graph()
    if snapshot == "kqapro-v1":
        path = Path(
            os.environ.get("GRAPHTASK_KQAPRO_DB", "data/processed/kqapro/kqapro-v1/graph.sqlite")
        )
        return SQLiteGraphBackend(path, snapshot_id=snapshot)
    if snapshot == "freebase-v1":
        endpoint = os.environ.get("FREEBASE_ENDPOINT", "")
        if not endpoint:
            raise ValueError("set FREEBASE_ENDPOINT for snapshot freebase-v1")
        return VirtuosoBackend(
            endpoint,
            timeout_s=float(os.environ.get("GRAPHTASK_GRAPH_TIMEOUT", "20")),
            retries=int(os.environ.get("GRAPHTASK_GRAPH_RETRIES", "2")),
            cache_path=Path(os.environ.get("GRAPHTASK_GRAPH_CACHE", "data/cache/virtuoso.sqlite")),
            snapshot_id=snapshot,
        )
    if snapshot.startswith("virtuoso:"):
        endpoint = snapshot.removeprefix("virtuoso:") or os.environ.get("FREEBASE_ENDPOINT", "")
        if not endpoint:
            raise ValueError("set FREEBASE_ENDPOINT or include it in the virtuoso snapshot URI")
        return VirtuosoBackend(
            endpoint,
            timeout_s=float(os.environ.get("GRAPHTASK_GRAPH_TIMEOUT", "20")),
            retries=int(os.environ.get("GRAPHTASK_GRAPH_RETRIES", "2")),
            cache_path=Path(os.environ.get("GRAPHTASK_GRAPH_CACHE", "data/cache/virtuoso.sqlite")),
            snapshot_id="legacy-virtuoso",
        )
    raise ValueError(f"unknown graph snapshot: {snapshot}")
