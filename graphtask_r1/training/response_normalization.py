from __future__ import annotations

import re

_THINK_PREFIX_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*",
    flags=re.DOTALL,
)


def normalize_graphscript_response(text: str) -> str:
    """Remove one optional leading Qwen thinking block before strict parsing."""

    return _THINK_PREFIX_PATTERN.sub("", text, count=1).strip()
