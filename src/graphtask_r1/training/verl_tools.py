"""Stateful tools for verl multi-turn rollout.

This module imports verl only when it is loaded by a verl worker. The rest of GraphTask-R1 does
not require verl to be installed.
"""

from __future__ import annotations

import json
from typing import Any

from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.schema import parse_program

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse
except ImportError as exc:  # pragma: no cover - exercised on the training server
    raise ImportError(
        "Install the pinned verl checkout before loading graphtask verl tools"
    ) from exc


class _SessionTool(BaseTool):  # type: ignore[misc]
    allowed_roles: frozenset[str] = frozenset()

    def __init__(self, config: dict[str, Any], tool_schema: Any) -> None:
        super().__init__(config, tool_schema)
        self._sessions: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: str | None = None, **kwargs: Any) -> tuple[str, Any]:
        instance_id, response = await super().create(instance_id)
        role = str(kwargs.get("role", ""))
        if role not in self.allowed_roles:
            raise ValueError(f"tool {self.name} is not available to role {role!r}")
        snapshot = str(kwargs.get("graph_snapshot", "toy-v1"))
        self._sessions[instance_id] = {
            "role": role,
            "backend": backend_from_snapshot(snapshot),
            "calls": 0,
            "task_id": kwargs.get("task_id"),
        }
        return instance_id, response

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        self._sessions.pop(instance_id, None)

    def _session(self, instance_id: str) -> dict[str, Any]:
        if instance_id not in self._sessions:
            raise KeyError(f"unknown tool session: {instance_id}")
        session = self._sessions[instance_id]
        session["calls"] += 1
        return session


class GraphSearchTool(_SessionTool):
    allowed_roles = frozenset({"questioner", "solver"})

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[Any, float, dict[str, float]]:
        del kwargs
        session = self._session(instance_id)
        triples = session["backend"].neighbors(
            [str(value) for value in parameters["entity_ids"]],
            direction=str(parameters.get("direction", "both")),
            relation_ids=[str(value) for value in parameters.get("relation_ids", [])] or None,
            limit=min(int(parameters.get("limit", 50)), int(self.config.get("max_results", 100))),
            trace_id=f"{session['task_id']}:{session['calls']}",
        )
        payload = [triple.model_dump(mode="json") for triple in triples]
        return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0.0, {"graph_calls": 1.0}


class InspectEntityTool(_SessionTool):
    allowed_roles = frozenset({"questioner", "solver"})

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[Any, float, dict[str, float]]:
        del kwargs
        session = self._session(instance_id)
        info = session["backend"].entity_info(str(parameters["entity_id"]))
        return ToolResponse(text=info.model_dump_json()), 0.0, {"graph_calls": 1.0}


class ExecuteProgramTool(_SessionTool):
    allowed_roles = frozenset({"questioner"})

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[Any, float, dict[str, float]]:
        del kwargs
        session = self._session(instance_id)
        program = parse_program(parameters["program"])
        answers = session["backend"].execute_program(program)
        response = {
            "cardinality": len(answers.answers),
            "answer_type": "count"
            if answers.answers and answers.answers[0].kind == "count"
            else "entity_set",
        }
        return ToolResponse(text=json.dumps(response)), 0.0, {"program_executions": 1.0}
