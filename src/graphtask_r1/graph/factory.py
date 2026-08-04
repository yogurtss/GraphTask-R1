from __future__ import annotations

import os
from pathlib import Path

from graphtask_r1.graph.base import GraphBackend
from graphtask_r1.graph.memory import toy_graph
from graphtask_r1.graph.virtuoso import VirtuosoBackend


def backend_from_snapshot(snapshot: str) -> GraphBackend:
    if snapshot == "toy-v1":
        return toy_graph()
    if snapshot.startswith("virtuoso:"):
        endpoint = snapshot.removeprefix("virtuoso:") or os.environ.get("FREEBASE_ENDPOINT", "")
        if not endpoint:
            raise ValueError("set FREEBASE_ENDPOINT or include it in the virtuoso snapshot URI")
        return VirtuosoBackend(
            endpoint,
            timeout_s=float(os.environ.get("GRAPHTASK_GRAPH_TIMEOUT", "20")),
            retries=int(os.environ.get("GRAPHTASK_GRAPH_RETRIES", "2")),
            cache_path=Path(os.environ.get("GRAPHTASK_GRAPH_CACHE", "data/cache/virtuoso.sqlite")),
        )
    raise ValueError(f"unknown graph snapshot: {snapshot}")
