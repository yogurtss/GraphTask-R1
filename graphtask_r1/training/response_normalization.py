from __future__ import annotations

import json
import re

_THINK_PREFIX_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*",
    flags=re.DOTALL,
)


def _extract_graphscript_json(text: str) -> str | None:
    """Return the first complete JSON object shaped like GraphScript."""

    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        object_start = text.find("{", search_from)
        if object_start < 0:
            return None
        try:
            payload, object_end = decoder.raw_decode(text, object_start)
        except json.JSONDecodeError:
            search_from = object_start + 1
            continue
        if (
            isinstance(payload, dict)
            and "version" in payload
            and isinstance(payload.get("ops"), list)
        ):
            return text[object_start:object_end]
        # Continue one character later so nested GraphScript objects can also
        # be discovered inside an unrelated outer object.
        search_from = object_start + 1


def normalize_graphscript_response(text: str) -> str:
    """Extract GraphScript JSON from a possibly wrapped model response.

    Serving stacks and models may prepend thinking text, tool-call delimiters,
    markdown, or other diagnostics.  The schema parser remains responsible for
    validating the extracted object's version and operations.
    """

    normalized = _THINK_PREFIX_PATTERN.sub("", text, count=1).strip()
    return _extract_graphscript_json(normalized) or normalized
