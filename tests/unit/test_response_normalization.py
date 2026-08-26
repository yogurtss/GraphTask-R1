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
        (
            '</tool_call>{"version":"0.3","ops":[]}',
            '{"version":"0.3","ops":[]}',
        ),
        (
            '<think>reasoning</think>\n</tool_call>  {"version":"0.3","ops":[]}',
            '{"version":"0.3","ops":[]}',
        ),
        (
            'arbitrary prefix <tag> noise {"version":"0.3","ops":[]} trailing text',
            '{"version":"0.3","ops":[]}',
        ),
        (
            '```json\n{"version":"0.3","ops":[]}\n```',
            '{"version":"0.3","ops":[]}',
        ),
        (
            'metadata={"request":1}; result={"version":"0.3","ops":[]}',
            '{"version":"0.3","ops":[]}',
        ),
        (
            '{"wrapper":{"version":"0.3","ops":[]}}',
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


@pytest.mark.parametrize(
    "response",
    [
        "</tool_call>",
        "</tool_call>not-json",
        '<tool_call>{"version":"0.3"}</tool_call>',
        '</tool_call>[{"version":"0.3"}]',
        'prefix {"version":"0.3","ops":"not-a-list"}',
        'prefix {"ops":[]}',
        'prefix {"version":"0.3","ops":[}',
    ],
)
def test_non_graphscript_content_is_not_extracted(response: str) -> None:
    assert normalize_graphscript_response(response) == response
