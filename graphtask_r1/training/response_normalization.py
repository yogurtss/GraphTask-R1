from __future__ import annotations

import json
import re

_THINK_PREFIX_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*",
    flags=re.DOTALL,
)


def _is_graphscript_payload(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and "version" in payload
        and isinstance(payload.get("ops"), list)
    )


def _is_questioner_envelope(payload: object) -> bool:
    """Recognize the Questioner contract without discarding its question."""

    return isinstance(payload, dict) and _is_graphscript_payload(payload.get("program"))


def _extract_structured_response_json(text: str) -> str | None:
    """Return the first complete Solver payload or Questioner envelope."""

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
        # Check the outer Questioner envelope before looking at nested objects.
        # Otherwise scanning would return only its GraphScript ``program`` and
        # silently drop the generated question used by Questioner rewards.
        if _is_questioner_envelope(payload) or _is_graphscript_payload(payload):
            return text[object_start:object_end]
        # Continue one character later so structured responses nested inside an
        # unrelated wrapper can still be discovered.
        search_from = object_start + 1


def normalize_reward_response(text: str) -> str:
    """Remove only the framework-injected thinking prefix for strict rewards.

    Training rewards must continue to see model-generated wrappers, prose, and
    suffixes so format violations are penalized instead of silently repaired.
    """

    return _THINK_PREFIX_PATTERN.sub("", text, count=1).strip()


def normalize_graphscript_response(text: str) -> str:
    """Extract structured JSON from a possibly wrapped model response.

    Serving stacks and models may prepend thinking text, tool-call delimiters,
    markdown, or other diagnostics. Solver GraphScript objects are extracted
    directly, while Questioner question/program envelopes remain intact. The
    role-specific parser remains responsible for full schema validation.
    """

    normalized = normalize_reward_response(text)
    return _extract_structured_response_json(normalized) or normalized
