from __future__ import annotations

import asyncio
from typing import Any

import pytest

from graphtask_r1.training.opponent import (
    OpponentUnavailable,
    _read_json_response,
)


class _Response:
    def __init__(self, body: str, *, status: int = 200, content_type: str = "text/plain"):
        self._body = body
        self.status = status
        self.headers = {"Content-Type": content_type}

    async def text(self) -> str:
        return self._body


def test_json_response_accepts_valid_json_served_as_text_plain() -> None:
    response: Any = _Response('{"choices": [{"message": {"content": "ok"}}]}')

    body = asyncio.run(_read_json_response(response, endpoint="test endpoint"))

    assert body["choices"][0]["message"]["content"] == "ok"


def test_non_json_response_becomes_retryable_opponent_unavailable() -> None:
    response: Any = _Response("upstream temporarily unavailable", status=502)

    with pytest.raises(OpponentUnavailable, match="status=502.*text/plain"):
        asyncio.run(_read_json_response(response, endpoint="test endpoint"))
