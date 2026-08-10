"""Stateful tools for verl multi-turn rollout.

This module imports verl only when it is loaded by a verl worker. The rest of GraphTask-R1 does
not require verl to be installed.
"""

from __future__ import annotations

import json
from typing import Any

from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.graphscript import execute_graphscript, program_to_graphscript
from graphtask_r1.schema import parse_program

try:
    from verl.tools.base_tool import BaseTool
except ImportError as exc:  # pragma: no cover - exercised on the training server
    raise ImportError(
        "Install the pinned verl checkout before loading graphtask verl tools"
    ) from exc

try:
    from verl.tools.schemas import ToolResponse as _ToolResponse
except ImportError:  # verl v0.5 returns plain strings from tools
    _ToolResponse = None  # type: ignore[assignment,misc]


def _tool_response(text: str) -> Any:
    """Return the response shape expected by the installed verl tool API."""
    if _ToolResponse is None:
        return text
    return _ToolResponse(text=text)


class _SessionTool(BaseTool):  # type: ignore[misc]
    allowed_roles: frozenset[str] = frozenset()

    def __init__(self, config: dict[str, Any], tool_schema: Any) -> None:
        super().__init__(config, tool_schema)
        self._sessions: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: str | None = None, **kwargs: Any) -> Any:
        created = await super().create(instance_id)
        resolved_instance_id = str(created[0] if isinstance(created, tuple) else created)
        role = str(kwargs.get("role", ""))
        if role not in self.allowed_roles:
            raise ValueError(f"tool {self.name} is not available to role {role!r}")
        snapshot = str(kwargs.get("graph_snapshot", "toy-v1"))
        topic_entity_ids = frozenset(
            str(value) for value in kwargs.get("topic_entity_ids", [])
        )
        self._sessions[resolved_instance_id] = {
            "role": role,
            "backend": backend_from_snapshot(snapshot),
            "calls": 0,
            "edge_visits": 0,
            "task_id": kwargs.get("task_id"),
            "allowed_relations": frozenset(
                str(value) for value in kwargs.get("allowed_relations", [])
            ),
            "max_follow_limit": int(kwargs.get("max_follow_limit", 100)),
            "max_edge_visits": int(kwargs.get("max_edge_visits", 200)),
            "max_returned_entities": int(kwargs.get("max_returned_entities", 1_000)),
            "program_profile": str(kwargs.get("program_profile", "full")),
            "topic_entity_ids": topic_entity_ids,
            "visible_entities": set(topic_entity_ids),
        }
        return created

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
        entity_ids = [str(value) for value in parameters["entity_ids"]]
        requested_relations = [str(value) for value in parameters.get("relation_ids", [])]
        allowed_relations = session["allowed_relations"]
        restricted = session["program_profile"] == "graphscript_v0_1"
        if restricted and (
            not entity_ids or not set(entity_ids).issubset(session["visible_entities"])
        ):
            raise ValueError("graph search must expand the seed or an observed entity")
        if restricted and allowed_relations:
            if requested_relations and not set(requested_relations).issubset(allowed_relations):
                raise ValueError("requested relation is outside the comparison catalog")
            requested_relations = requested_relations or sorted(allowed_relations)
        limit = min(int(parameters.get("limit", 50)), int(self.config.get("max_results", 100)))
        if restricted:
            role = str(session["role"])
            max_calls = 7 if role == "questioner" else 8
            if int(session["calls"]) > max_calls:
                raise ValueError("graph-search call budget exceeded")
            search_budget = int(session["max_edge_visits"])
            if role == "questioner":
                search_budget //= 2
            remaining = search_budget - int(session["edge_visits"])
            if remaining <= 0:
                raise ValueError("graph-search edge budget exhausted")
            limit = min(limit, remaining + 1)
        triples = session["backend"].neighbors(
            entity_ids,
            direction=str(parameters.get("direction", "both")),
            relation_ids=requested_relations or None,
            limit=limit,
            trace_id=f"{session['task_id']}:{session['calls']}",
        )
        if restricted:
            session["edge_visits"] += len(triples)
            session["visible_entities"].update(
                value
                for triple in triples
                for value in (triple.subject, triple.object)
            )
            role = str(session["role"])
            search_budget = int(session["max_edge_visits"])
            if role == "questioner":
                search_budget //= 2
            if int(session["edge_visits"]) > search_budget:
                raise ValueError("graph-search edge budget exceeded")
        payload = [triple.model_dump(mode="json") for triple in triples]
        return _tool_response(json.dumps(payload, ensure_ascii=False)), 0.0, {
            "graph_calls": 1.0,
            "edge_visits": float(len(triples)),
        }


class InspectEntityTool(_SessionTool):
    allowed_roles = frozenset({"questioner", "solver"})

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[Any, float, dict[str, float]]:
        del kwargs
        session = self._session(instance_id)
        info = session["backend"].entity_info(str(parameters["entity_id"]))
        return _tool_response(info.model_dump_json()), 0.0, {"graph_calls": 1.0}


class ExecuteProgramTool(_SessionTool):
    allowed_roles = frozenset({"questioner"})

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[Any, float, dict[str, float]]:
        del kwargs
        session = self._session(instance_id)
        program = parse_program(parameters["program"])
        usage = {"program_executions": 1.0}
        if session["program_profile"] == "graphscript_v0_1":
            if int(session["calls"]) > 1:
                raise ValueError("comparison profile permits one candidate program execution")
            if _program_seed(program) != session["topic_entity_ids"]:
                raise ValueError("comparison program must be rooted at the episode topic entity")
            script = program_to_graphscript(
                program, follow_limit=int(session["max_follow_limit"])
            )
            execution = execute_graphscript(
                script,
                session["backend"],
                seed_entity=next(iter(session["topic_entity_ids"])),
                allowed_relations=session["allowed_relations"],
                max_edge_visits=max(1, int(session["max_edge_visits"]) // 2),
                max_returned_entities=int(session["max_returned_entities"]),
                trace_id=f"{session['task_id']}:{session['calls']}",
            )
            answers = execution.answers
            usage.update(
                {
                    "edge_visits": float(execution.usage.edge_visits),
                    "graph_calls": float(execution.usage.graph_calls),
                }
            )
        else:
            answers = session["backend"].execute_program(program)
        response = {
            "cardinality": len(answers.answers),
            "answer_type": "count"
            if answers.answers and answers.answers[0].kind == "count"
            else "entity_set",
        }
        return _tool_response(json.dumps(response)), 0.0, usage


def _program_seed(program: Any) -> frozenset[str]:
    from graphtask_r1.schema import Entity, Hop

    node = program
    while isinstance(node, Hop):
        node = node.input
    if not isinstance(node, Entity):
        raise ValueError("comparison programs must have one entity root")
    return frozenset({node.entity_id})
