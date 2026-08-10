from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from graphtask_r1.training.json_compat import to_json_compatible

GRAPH_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "graph_search",
        "description": "Traverse real graph edges adjacent to one or more entities.",
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
            },
            "required": ["entity_ids", "direction"],
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


def tool_schemas(role: str) -> list[dict[str, object]]:
    """Return a fresh, JSON-native tool list for one GraphTask role."""
    tools = [GRAPH_SEARCH_TOOL, INSPECT_ENTITY_TOOL]
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
        raise ValueError("messages must be a list")
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
    return {
        "messages": normalize_agent_messages(normalized.get("messages")),
        "tools": json.dumps(tool_schemas(role), ensure_ascii=False, separators=(",", ":")),
        "role": role,
        "task_id": str(normalized.get("task_id", "")),
        "interaction_mode": str(normalized.get("interaction_mode", "tool")),
    }


def convert_rl_row(row: Mapping[str, object]) -> dict[str, object]:
    """Adapt one existing verl RL row for ms-swift at load time."""
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
    ground_truth = str(reward_model.get("ground_truth", ""))
    return {
        "messages": normalize_agent_messages(normalized.get("prompt")),
        "tools": json.dumps(tool_schemas(role), ensure_ascii=False, separators=(",", ":")),
        "data_source": str(normalized.get("data_source", "graphtask/solver")),
        "ground_truth": ground_truth,
        "solution": ground_truth,
        "extra_info": dict(extra_info),
        "uid": str(normalized.get("uid", "")),
    }
