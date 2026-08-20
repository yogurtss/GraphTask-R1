from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from graphtask_r1.envs.graph_query import COMPACT_GRAPH_QUERY_SCHEMA
from graphtask_r1.training.json_compat import to_json_compatible

GRAPH_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "graph_search",
        "description": (
            "Traverse graph edges, or execute a bounded traversal/filter query without "
            "materializing large intermediate entity lists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact entity IDs to expand.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["out", "in", "both"],
                    "description": "Edge direction.",
                },
                "relation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional exact relation IDs.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum triples returned.",
                },
                "query": COMPACT_GRAPH_QUERY_SCHEMA,
            },
            "anyOf": [
                {"required": ["entity_ids", "direction"]},
                {"required": ["query"]},
            ],
        },
    },
}

INSPECT_ENTITY_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_entity",
        "description": "Return the label, aliases, and types for an exact entity ID.",
        "parameters": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
}

TEXT_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "text_search",
        "description": (
            "Search the indexed Wikipedia passages when a question has no exact topic entity ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum passage hits; defaults to 3.",
                },
            },
            "required": ["query"],
        },
    },
}

EXECUTE_PROGRAM_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_program",
        "description": "Questioner-only execution of a typed candidate program.",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "object",
                    "description": "GraphTask typed program JSON.",
                }
            },
            "required": ["program"],
        },
    },
}


def tool_schemas(role: str, *, text_search_enabled: bool = False) -> list[dict[str, object]]:
    """Return a fresh, JSON-native tool list for one GraphTask role."""
    tools = [GRAPH_SEARCH_TOOL, INSPECT_ENTITY_TOOL]
    if role == "solver" and text_search_enabled:
        tools.append(TEXT_SEARCH_TOOL)
    if role == "questioner":
        tools.append(EXECUTE_PROGRAM_TOOL)
    return cast(list[dict[str, object]], to_json_compatible(tools))


def _arguments(value: object) -> object:
    normalized = to_json_compatible(value)
    if not isinstance(normalized, str):
        return normalized
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        return normalized


def _tool_call_message(call: Mapping[str, object]) -> dict[str, str]:
    function = call.get("function", {})
    if not isinstance(function, Mapping):
        raise ValueError("tool call function must be an object")
    name = str(function.get("name", "")).strip()
    if not name:
        raise ValueError("tool call function name is required")
    content = {"name": name, "arguments": _arguments(function.get("arguments", {}))}
    return {
        "role": "tool_call",
        "content": json.dumps(content, ensure_ascii=False, separators=(",", ":")),
    }


def normalize_agent_messages(messages: object) -> list[dict[str, str]]:
    """Convert existing OpenAI-style Parquet messages to ms-swift agent messages."""
    normalized = to_json_compatible(messages)
    if not isinstance(normalized, list):
        raise ValueError(
            "messages must be a list, got "
            f"{type(normalized).__name__}; check that TRAIN_DATA/VAL_DATA point to "
            "preflight SFT accepted Parquet files, not certified tasks or RL rows"
        )
    result: list[dict[str, str]] = []
    for raw_message in normalized:
        if not isinstance(raw_message, Mapping):
            raise ValueError("each message must be an object")
        role = str(raw_message.get("role", ""))
        content = raw_message.get("content", "")
        if role == "assistant" and raw_message.get("tool_calls"):
            if content:
                result.append({"role": "assistant", "content": str(content)})
            calls = raw_message["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be a list")
            result.extend(_tool_call_message(call) for call in calls)
        elif role in {"tool", "tool_response"}:
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            result.append({"role": "tool_response", "content": content})
        elif role in {"system", "user", "assistant", "tool_call"}:
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            result.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported message role: {role!r}")
    if not result:
        raise ValueError("messages cannot be empty")
    return result


def convert_sft_row(row: Mapping[str, object]) -> dict[str, object]:
    """Adapt one existing SFT row without writing or regenerating a dataset."""
    normalized = to_json_compatible(row)
    if not isinstance(normalized, dict):
        raise ValueError("SFT row must be an object")
    role = str(normalized.get("role", "solver"))
    text_search_enabled = bool(normalized.get("text_search_enabled", False))
    interaction_mode = str(normalized.get("interaction_mode", "tool"))
    result: dict[str, object] = {
        "messages": normalize_agent_messages(normalized.get("messages")),
        "role": role,
        "task_id": str(normalized.get("task_id", "")),
        "interaction_mode": interaction_mode,
    }
    if interaction_mode == "tool":
        result["tools"] = json.dumps(
            tool_schemas(role, text_search_enabled=text_search_enabled),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return result


def convert_rl_row(row: Mapping[str, object]) -> dict[str, object]:
    """Adapt one GraphTask RL row for ms-swift at load time."""
    normalized = to_json_compatible(row)
    if not isinstance(normalized, dict):
        raise ValueError("RL row must be an object")
    reward_model = normalized.get("reward_model", {})
    if not isinstance(reward_model, Mapping):
        raise ValueError("reward_model must be an object")
    extra_info = normalized.get("extra_info", {})
    if not isinstance(extra_info, Mapping):
        raise ValueError("extra_info must be an object")
    role = str(extra_info.get("role", "solver"))
    text_search_enabled = bool(extra_info.get("text_search_enabled", False))
    interaction_mode = str(extra_info.get("interaction_mode", "tool"))
    ground_truth = str(reward_model.get("ground_truth", ""))
    result: dict[str, object] = {
        "messages": normalize_agent_messages(normalized.get("prompt")),
        "data_source": str(normalized.get("data_source", "graphtask/solver")),
        "ground_truth": ground_truth,
        "solution": ground_truth,
        "extra_info": dict(extra_info),
        "uid": str(normalized.get("uid", "")),
    }
    if interaction_mode == "tool":
        result["tools"] = json.dumps(
            tool_schemas(role, text_search_enabled=text_search_enabled),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return result
