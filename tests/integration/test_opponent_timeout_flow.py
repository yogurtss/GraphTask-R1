from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast

import pytest

from graphtask_r1.schema import Entity, Hop, TaskProposal
from graphtask_r1.training.opponent import (
    FrozenSolverService,
    OpponentTimeout,
    create_app,
    request_opponent,
)


class _SlowToyOpponent(FrozenSolverService):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(
            model_url="http://unused",
            model="slow-toy-opponent",
            archive_path=tmp_path / "archive.sqlite",
            candidate_archive_path=tmp_path / "candidates.sqlite",
            cache_evaluations=True,
        )
        self.delay_s = 2.0
        self.cancelled_samples: list[int] = []

    async def rollout(self, task: Any, backend: Any, **kwargs: Any) -> dict[str, float]:
        del task, backend
        sample_index = int(kwargs["sample_index"])
        try:
            await asyncio.sleep(self.delay_s)
        except asyncio.CancelledError:
            self.cancelled_samples.append(sample_index)
            raise
        return {
            "passed": 1.0,
            "f1": 1.0,
            "tool_calls": 0.0,
            "invalid_tool_calls": 0.0,
            "edge_visits": 0.0,
            "program_parse": 1.0,
            "program_executable": 1.0,
            "program_operators": 2.0,
            "passage_searches": 0.0,
            "latency_ms": self.delay_s * 1_000,
        }


def _proposal() -> TaskProposal:
    return TaskProposal(
        topic_entities=("alice",),
        program=Hop(
            input=Hop(input=Entity(entity_id="alice"), relation="works_at"),
            relation="located_in",
        ),
    )


def test_slow_toy_opponent_times_out_without_retry_or_late_archive(
    tmp_path: Path,
) -> None:
    from aiohttp import web

    service = _SlowToyOpponent(tmp_path)

    async def run() -> tuple[OpponentTimeout, float, dict[str, Any]]:
        runner = web.AppRunner(create_app(service))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        try:
            await site.start()
        except OSError as exc:
            await runner.cleanup()
            pytest.skip(f"loopback sockets are unavailable in this environment: {exc}")
        server = cast(Any, site)._server
        assert server is not None and server.sockets
        port = int(server.sockets[0].getsockname()[1])
        url = f"http://127.0.0.1:{port}"
        try:
            started = time.perf_counter()
            with pytest.raises(OpponentTimeout) as raised:
                await request_opponent(
                    url,
                    proposal=_proposal(),
                    graph_snapshot="toy-v1",
                    samples=3,
                    round_index=2,
                    timeout_s=0.5,
                    retries=4,
                    seed=17,
                    trace_id="integration:slow-toy",
                )
            elapsed = time.perf_counter() - started
            assert service._evaluation_cache == {}
            assert service.candidate_archive is not None
            assert service.candidate_archive.all() == []

            service.delay_s = 0.0
            recovered = await request_opponent(
                url,
                proposal=_proposal(),
                graph_snapshot="toy-v1",
                samples=3,
                round_index=2,
                timeout_s=1.0,
                retries=0,
                seed=17,
                trace_id="integration:recovered-toy",
            )
            assert len(service.candidate_archive.all()) == 1
            return raised.value, elapsed, recovered
        finally:
            await runner.cleanup()

    error, elapsed, recovered = asyncio.run(run())

    assert error.stage == "server_evaluation"
    assert error.attempts == 1
    assert error.trace_id == "integration:slow-toy"
    assert elapsed < 0.75
    assert sorted(service.cancelled_samples) == [0, 1, 2]
    assert recovered["samples"] == 3
    assert recovered["pass_rate"] == 1.0
