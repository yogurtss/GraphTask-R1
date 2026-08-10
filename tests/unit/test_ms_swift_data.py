from __future__ import annotations

import json

import numpy as np

from graphtask_r1.training.ms_swift_data import convert_rl_row, convert_sft_row


def _messages() -> np.ndarray:
    return np.array(
        [
            {"role": "system", "content": "Use graph tools."},
            {"role": "user", "content": "Who works here?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": np.array(
                    [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "graph_search",
                                "arguments": {
                                    "entity_ids": np.array(["alice"], dtype=object),
                                    "direction": "out",
                                },
                            },
                        }
                    ],
                    dtype=object,
                ),
            },
            {"role": "tool", "content": '{"triples":[]}', "tool_call_id": "call-1"},
            {"role": "assistant", "content": '<answer>["alice"]</answer>'},
        ],
        dtype=object,
    )


def test_sft_adapter_converts_existing_openai_tool_messages() -> None:
    converted = convert_sft_row(
        {
            "messages": _messages(),
            "role": "solver",
            "task_id": "task-1",
            "interaction_mode": "tool",
        }
    )

    messages = converted["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "tool_call",
        "tool_response",
        "assistant",
    ]
    assert json.loads(messages[2]["content"]) == {
        "name": "graph_search",
        "arguments": {"entity_ids": ["alice"], "direction": "out"},
    }
    tools = json.loads(str(converted["tools"]))
    assert [tool["function"]["name"] for tool in tools] == [
        "graph_search",
        "inspect_entity",
    ]
    json.dumps(converted)


def test_rl_adapter_reuses_prompt_and_reward_columns_in_memory() -> None:
    converted = convert_rl_row(
        {
            "data_source": "graphtask/solver",
            "prompt": np.array(
                [
                    {"role": "system", "content": "Use graph tools."},
                    {"role": "user", "content": "Question"},
                ],
                dtype=object,
            ),
            "reward_model": {"ground_truth": '{"answers":[]}'},
            "extra_info": {
                "role": "solver",
                "graph_snapshot": "kqapro-v1",
                "topic_entity_ids": np.array(["alice"], dtype=object),
            },
            "uid": "solver:task-1",
        }
    )

    assert converted["ground_truth"] == '{"answers":[]}'
    assert converted["solution"] == converted["ground_truth"]
    assert converted["extra_info"] == {
        "role": "solver",
        "graph_snapshot": "kqapro-v1",
        "topic_entity_ids": ["alice"],
    }
    json.dumps(converted)
