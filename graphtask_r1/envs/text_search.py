from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import PassageHit

MAX_TEXT_SEARCH_RESULTS = 10
MAX_PASSAGE_CHARS = 4_000


@runtime_checkable
class TextSearchBackend(Protocol):
    def search_text(
        self,
        query: str,
        *,
        limit: int = 3,
        max_chars: int = 2_000,
        trace_id: str | None = None,
    ) -> list[dict[str, Any]]: ...


def execute_text_search(
    backend: GraphBackend,
    query: str,
    *,
    limit: int = 3,
    max_chars: int = 2_000,
    trace_id: str | None = None,
) -> tuple[PassageHit, ...]:
    """Run a bounded passage search on a backend with an indexed text sidecar."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("text-search query cannot be empty")
    if not 1 <= limit <= MAX_TEXT_SEARCH_RESULTS:
        raise ValueError(f"text-search limit must be between 1 and {MAX_TEXT_SEARCH_RESULTS}")
    if not 1 <= max_chars <= MAX_PASSAGE_CHARS:
        raise ValueError(f"max_chars must be between 1 and {MAX_PASSAGE_CHARS}")
    if not isinstance(backend, TextSearchBackend):
        raise ValueError("current graph snapshot has no passage text index")
    rows = backend.search_text(
        normalized_query,
        limit=limit,
        max_chars=max_chars,
        trace_id=trace_id,
    )
    return tuple(PassageHit.model_validate(row) for row in rows)
