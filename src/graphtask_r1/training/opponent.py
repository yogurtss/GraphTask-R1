from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from graphtask_r1.archive import TaskArchive
from graphtask_r1.evaluation import answer_metrics
from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import GraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import execute_graphscript, parse_graphscript
from graphtask_r1.schema import BenchmarkExample, RelationInfo, TaskProposal
from graphtask_r1.training.parsing import parse_solver_output
from graphtask_r1.training.prompts import InteractionMode, role_prompt
from graphtask_r1.training.relations import load_relation_catalog

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": "Traverse graph edges adjacent to one or more entity IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_ids": {"type": "array", "items": {"type": "string"}},
                    "direction": {"type": "string", "enum": ["out", "in", "both"]},
                    "relation_ids": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                },
                "required": ["entity_ids", "direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_entity",
            "description": "Return label, aliases, and types for an entity ID.",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        },
    },
]


class OpponentUnavailable(RuntimeError):
    pass


async def request_opponent(
    url: str,
    *,
    proposal: TaskProposal,
    graph_snapshot: str,
    samples: int,
    round_index: int | None,
    timeout_s: float = 180.0,
    retries: int = 2,
    interaction_mode: InteractionMode = "tool",
    allowed_relations: tuple[str, ...] = (),
    max_follow_limit: int = 100,
    max_edge_visits: int | None = None,
) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - training extra
        raise ImportError("install graphtask-r1[training] for async opponent rewards") from exc
    payload = {
        "proposal": proposal.model_dump(mode="json"),
        "graph_snapshot": graph_snapshot,
        "samples": samples,
        "round": round_index,
        "interaction_mode": interaction_mode,
        "allowed_relations": list(allowed_relations),
        "max_follow_limit": max_follow_limit,
    }
    if max_edge_visits is not None:
        payload["max_edge_visits"] = max_edge_visits
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url.rstrip("/") + "/evaluate", json=payload) as response,
            ):
                body = await response.json()
                if response.status != 200:
                    raise OpponentUnavailable(str(body))
                return dict(body)
        except (aiohttp.ClientError, TimeoutError, ValueError, OpponentUnavailable) as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(min(0.5 * 2**attempt, 2.0))
    raise OpponentUnavailable(f"opponent unavailable after {retries + 1} attempts") from last_error


class FrozenSolverService:
    def __init__(
        self,
        *,
        model_url: str,
        model: str,
        archive_path: Path,
        max_turns: int = 8,
        request_timeout_s: float = 120.0,
        interaction_mode: InteractionMode = "tool",
        relation_catalog: tuple[RelationInfo, ...] = (),
        max_follow_limit: int = 100,
        max_edge_visits: int | None = None,
    ) -> None:
        self.model_url = model_url.rstrip("/")
        self.model = model
        self.archive = TaskArchive(archive_path)
        self.max_turns = max_turns
        self.request_timeout_s = request_timeout_s
        self.interaction_mode = interaction_mode
        self.relation_catalog = relation_catalog
        self.max_follow_limit = max_follow_limit
        self.max_edge_visits = max_edge_visits
        self.backends: dict[str, GraphBackend] = {}
        if interaction_mode == "graphscript" and not relation_catalog:
            raise ValueError("graphscript opponent requires a non-empty relation catalog")

    def backend(self, snapshot: str) -> GraphBackend:
        if snapshot not in self.backends:
            self.backends[snapshot] = backend_from_snapshot(snapshot)
        return self.backends[snapshot]

    async def _completion(
        self, messages: list[dict[str, Any]], *, use_tools: bool
    ) -> dict[str, Any]:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 2048,
        }
        if use_tools:
            payload["tools"] = TOOLS
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(self.model_url + "/v1/chat/completions", json=payload) as response,
        ):
            body = await response.json()
            if response.status != 200:
                raise OpponentUnavailable(f"SGLang returned {response.status}: {body}")
            return dict(body["choices"][0]["message"])

    @staticmethod
    def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
        value = tool_call["function"].get("arguments", {})
        return dict(json.loads(value) if isinstance(value, str) else value)

    def _execute_tool(
        self,
        backend: GraphBackend,
        name: str,
        arguments: dict[str, Any],
        *,
        allowed_relations: frozenset[str],
        remaining_edge_visits: int | None,
        visible_entities: set[str],
        restrict_frontier: bool,
    ) -> tuple[str, int]:
        if name == "graph_search":
            entity_ids = [str(value) for value in arguments["entity_ids"]]
            if restrict_frontier and (
                not entity_ids or not set(entity_ids).issubset(visible_entities)
            ):
                raise ValueError("opponent must expand the seed or an observed entity")
            relation_ids = [str(value) for value in arguments.get("relation_ids", [])]
            if allowed_relations:
                if relation_ids and not set(relation_ids).issubset(allowed_relations):
                    raise ValueError("opponent requested a relation outside the shared catalog")
                relation_ids = relation_ids or sorted(allowed_relations)
            limit = min(int(arguments.get("limit", 50)), 100)
            if remaining_edge_visits is not None:
                limit = min(limit, remaining_edge_visits + 1)
            triples = backend.neighbors(
                entity_ids,
                direction=str(arguments.get("direction", "both")),
                relation_ids=relation_ids or None,
                limit=limit,
            )
            if restrict_frontier:
                visible_entities.update(
                    value
                    for triple in triples
                    for value in (triple.subject, triple.object)
                )
            return json.dumps([value.model_dump(mode="json") for value in triples]), len(triples)
        if name == "inspect_entity":
            entity_id = str(arguments["entity_id"])
            if restrict_frontier and entity_id not in visible_entities:
                raise ValueError("opponent may inspect only the seed or an observed entity")
            return backend.entity_info(entity_id).model_dump_json(), 0
        raise ValueError(f"opponent requested unsupported tool: {name}")

    async def rollout(
        self,
        task: Any,
        backend: GraphBackend,
        *,
        interaction_mode: InteractionMode | None = None,
        allowed_relations: tuple[str, ...] = (),
        max_follow_limit: int | None = None,
        max_edge_visits: int | None = None,
    ) -> dict[str, float]:
        mode = interaction_mode or self.interaction_mode
        catalog = self.relation_catalog
        configured_relations = {value.relation_id for value in self.relation_catalog}
        if allowed_relations:
            unknown = sorted(set(allowed_relations) - configured_relations)
            if configured_relations and unknown:
                raise ValueError(
                    "requested relations are outside the opponent catalog: "
                    + ", ".join(unknown)
                )
            labels = {value.relation_id: value for value in catalog}
            catalog = tuple(
                labels[value] if value in labels else backend.relation_info(value)
                for value in allowed_relations
            )
        topic_ids = [entity.entity_id for entity in task.topic_entities]
        messages: list[dict[str, Any]] = list(
            role_prompt(
                "solver",
                f"Question: {task.question}\nTopic entities: {', '.join(topic_ids)}",
                interaction_mode=mode,
                relation_catalog=catalog,
            )
        )
        tool_calls = 0
        edge_visits = 0
        visible_entities = set(topic_ids)
        started = time.perf_counter()
        if mode == "graphscript":
            if len(topic_ids) != 1:
                return {
                    "passed": 0.0,
                    "f1": 0.0,
                    "tool_calls": 0.0,
                    "edge_visits": 0.0,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            message = await self._completion(messages, use_tools=False)
            try:
                script = parse_graphscript(
                    str(message.get("content", "")),
                    max_follow_limit=max_follow_limit or self.max_follow_limit,
                )
                execution = execute_graphscript(
                    script,
                    backend,
                    seed_entity=topic_ids[0],
                    allowed_relations=frozenset(
                        allowed_relations
                        or (value.relation_id for value in self.relation_catalog)
                    ),
                    max_edge_visits=max_edge_visits or self.max_edge_visits or 200,
                    trace_id=str(getattr(task, "task_id", "opponent")),
                )
                metrics = answer_metrics(execution.answers, task.gold_answers)
                edge_visits = execution.usage.edge_visits
            except (TypeError, ValueError, json.JSONDecodeError):
                metrics = {"f1": 0.0, "exact_match": 0.0}
                edge_visits = 0
            return {
                "passed": float(metrics["exact_match"]),
                "f1": float(metrics["f1"]),
                "tool_calls": 0.0,
                "edge_visits": edge_visits,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        for _ in range(self.max_turns):
            message = await self._completion(messages, use_tools=True)
            messages.append(message)
            calls = message.get("tool_calls") or []
            if calls:
                for call in calls:
                    try:
                        name = str(call["function"]["name"])
                        edge_budget = max_edge_visits or self.max_edge_visits
                        effective_relations = frozenset(
                            allowed_relations
                            or (value.relation_id for value in self.relation_catalog)
                        )
                        result, visited = self._execute_tool(
                            backend,
                            name,
                            self._arguments(call),
                            allowed_relations=effective_relations,
                            remaining_edge_visits=edge_budget - edge_visits
                            if edge_budget is not None
                            else None,
                            visible_entities=visible_entities,
                            restrict_frontier=edge_budget is not None
                            and bool(effective_relations),
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        return {
                            "passed": 0.0,
                            "f1": 0.0,
                            "tool_calls": float(tool_calls),
                            "edge_visits": float(edge_visits),
                            "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                    edge_visits += visited
                    if edge_budget is not None and edge_visits > edge_budget:
                        return {
                            "passed": 0.0,
                            "f1": 0.0,
                            "tool_calls": float(tool_calls),
                            "edge_visits": float(edge_visits),
                            "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"call-{tool_calls}"),
                            "name": name,
                            "content": result,
                        }
                    )
                    tool_calls += 1
                continue
            content = str(message.get("content", ""))
            try:
                count = bool(
                    task.gold_answers.answers and task.gold_answers.answers[0].kind == "count"
                )
                predicted = parse_solver_output(content, count=count)
                metrics = answer_metrics(predicted, task.gold_answers)
            except (TypeError, ValueError, json.JSONDecodeError):
                metrics = {"f1": 0.0, "exact_match": 0.0}
            return {
                "passed": float(metrics["exact_match"]),
                "f1": float(metrics["f1"]),
                "tool_calls": float(tool_calls),
                "edge_visits": float(edge_visits),
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        return {
            "passed": 0.0,
            "f1": 0.0,
            "tool_calls": float(tool_calls),
            "edge_visits": float(edge_visits),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    async def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        proposal = TaskProposal.model_validate(payload["proposal"])
        snapshot = str(payload["graph_snapshot"])
        samples = max(1, min(int(payload.get("samples", 8)), 64))
        round_index = payload.get("round")
        raw_mode = str(payload.get("interaction_mode", self.interaction_mode))
        if raw_mode not in {"tool", "graphscript"}:
            raise ValueError(f"unsupported interaction mode: {raw_mode}")
        mode = cast(InteractionMode, raw_mode)
        allowed_relations = tuple(str(value) for value in payload.get("allowed_relations", []))
        backend = self.backend(snapshot)
        task = certify_proposal(proposal, backend, graph_snapshot=snapshot, round_index=round_index)
        structural, textual = self.archive.novelty(task.program_signature, task.question)
        results = await asyncio.gather(
            *(
                self.rollout(
                    task,
                    backend,
                    interaction_mode=mode,
                    allowed_relations=allowed_relations,
                    max_follow_limit=int(payload.get("max_follow_limit", self.max_follow_limit)),
                    max_edge_visits=int(payload["max_edge_visits"])
                    if payload.get("max_edge_visits") is not None
                    else self.max_edge_visits,
                )
                for _ in range(samples)
            )
        )
        summary = {
            "task_id": task.task_id,
            "pass_rate": sum(value["passed"] for value in results) / samples,
            "mean_f1": sum(value["f1"] for value in results) / samples,
            "mean_tool_calls": sum(value["tool_calls"] for value in results) / samples,
            "mean_edge_visits": sum(value.get("edge_visits", 0.0) for value in results) / samples,
            "novelty_structural": structural,
            "novelty_textual": textual,
            "samples": samples,
        }
        task = task.model_copy(
            update={
                "solver_stats": {
                    **task.solver_stats,
                    **summary,
                    "interaction_mode": mode,
                },
                "generation": {**task.generation, "interaction_mode": mode},
            }
        )
        self.archive.add(task)
        return summary

    async def solve(self, payload: dict[str, Any]) -> dict[str, Any]:
        example = BenchmarkExample.model_validate(payload["example"])
        snapshot = str(payload.get("graph_snapshot", "freebase-v1"))
        samples = max(1, min(int(payload.get("samples", 1)), 64))
        backend = self.backend(snapshot)
        task = SimpleNamespace(
            question=example.question,
            topic_entities=tuple(backend.entity_info(value) for value in example.topic_entity_ids),
            gold_answers=example.gold_answers,
        )
        results = await asyncio.gather(*(self.rollout(task, backend) for _ in range(samples)))
        return {
            "example_id": example.example_id,
            "pass_rate": sum(value["passed"] for value in results) / samples,
            "mean_f1": sum(value["f1"] for value in results) / samples,
            "mean_tool_calls": sum(value["tool_calls"] for value in results) / samples,
            "mean_edge_visits": sum(value.get("edge_visits", 0.0) for value in results) / samples,
            "mean_latency_ms": sum(value["latency_ms"] for value in results) / samples,
            "samples": samples,
        }


def create_app(service: FrozenSolverService) -> Any:
    from aiohttp import web

    async def health(_: Any) -> Any:
        return web.json_response({"status": "ok", "model": service.model})

    async def evaluate(request: Any) -> Any:
        try:
            return web.json_response(await service.evaluate(await request.json()))
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            return web.json_response({"error": type(exc).__name__, "detail": str(exc)}, status=422)

    async def solve(request: Any) -> Any:
        try:
            return web.json_response(await service.solve(await request.json()))
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            return web.json_response({"error": type(exc).__name__, "detail": str(exc)}, status=422)

    app = web.Application(client_max_size=2 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/evaluate", evaluate)
    app.router.add_post("/solve", solve)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(prog="graphtask-opponent")
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--interaction-mode", choices=["tool", "graphscript"], default="tool")
    parser.add_argument("--relation-catalog", type=Path)
    parser.add_argument("--max-follow-limit", type=int, default=100)
    parser.add_argument("--max-edge-visits", type=int)
    args = parser.parse_args()
    from aiohttp import web

    service = FrozenSolverService(
        model_url=args.model_url,
        model=args.model,
        archive_path=args.archive,
        max_turns=args.max_turns,
        interaction_mode=args.interaction_mode,
        relation_catalog=load_relation_catalog(args.relation_catalog),
        max_follow_limit=args.max_follow_limit,
        max_edge_visits=args.max_edge_visits,
    )
    web.run_app(create_app(service), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
