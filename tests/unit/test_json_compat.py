import json

import numpy as np

from graphtask_r1.training.json_compat import to_json_compatible


def test_nested_numpy_tool_call_is_json_serializable() -> None:
    messages = np.array(
        [
            {
                "role": "assistant",
                "tool_calls": np.array(
                    [
                        {
                            "function": {
                                "name": "graph_search",
                                "arguments": {
                                    "entity_ids": np.array(["alice"], dtype=object),
                                    "relation_ids": np.array(["works_at"], dtype=object),
                                },
                            }
                        }
                    ],
                    dtype=object,
                ),
            }
        ],
        dtype=object,
    )

    normalized = to_json_compatible(messages)
    encoded = json.dumps(normalized)
    assert json.loads(encoded)[0]["tool_calls"][0]["function"]["arguments"] == {
        "entity_ids": ["alice"],
        "relation_ids": ["works_at"],
    }
