from __future__ import annotations

import pytest

from graphtask_r1.training.response_normalization import normalize_graphscript_response


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"version":"0.3","ops":[]}', '{"version":"0.3","ops":[]}'),
        (
            '<think></think>{"version":"0.3","ops":[]}',
            '{"version":"0.3","ops":[]}',
        ),
        (
            '<think>\n\n</think>\n\n{"version":"0.3","ops":[]}',
            '{"version":"0.3","ops":[]}',
        ),
        (
            '<think>reasoning</think>{"version":"0.3","ops":[]}',
            '{"version":"0.3","ops":[]}',
        ),
    ],
)
def test_normalize_graphscript_response(raw: str, expected: str) -> None:
    assert normalize_graphscript_response(raw) == expected


def test_thinking_without_payload_remains_empty() -> None:
    assert normalize_graphscript_response("<think></think>") == ""


def test_only_one_leading_thinking_block_is_removed() -> None:
    response = '<think>first</think><think>second</think>{"version":"0.3"}'

    assert normalize_graphscript_response(response) == (
        '<think>second</think>{"version":"0.3"}'
    )
