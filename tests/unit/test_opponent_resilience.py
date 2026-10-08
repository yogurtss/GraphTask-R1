from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from graphtask_r1.schema import Entity, Hop, TaskProposal
from graphtask_r1.training import opponent as opponent_module
from graphtask_r1.training.opponent import (
    FrozenSolverService,
    OpponentTimeout,
    OpponentUnavailable,
    _read_json_response,
    request_opponent,
)


class _Response:
    def __init__(self, body: str, *, status: int = 200, content_type: str = "text/plain"):
        self._body = body
        self.status = status
        self.headers = {"Content-Type": content_type}

    async def text(self) -> str:
        return self._body


def test_json_response_accepts_valid_json_served_as_text_plain() -> None:
    response: Any = _Response('{"choices": [{"message": {"content": "ok"}}]}')

    body = asyncio.run(_read_json_response(response, endpoint="test endpoint"))

    assert body["choices"][0]["message"]["content"] == "ok"


def test_non_json_response_becomes_retryable_opponent_unavailable() -> None:
    response: Any = _Response("upstream temporarily unavailable", status=502)

    with pytest.raises(OpponentUnavailable, match="status=502.*text/plain"):
        asyncio.run(_read_json_response(response, endpoint="test endpoint"))


def _proposal() -> TaskProposal:
    return TaskProposal(
        topic_entities=("alice",),
        program=Hop(
            input=Hop(input=Entity(entity_id="alice"), relation="works_at"),
            relation="located_in",
        ),
    )


def test_request_opponent_uses_one_total_deadline_and_forwards_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiohttp

    payloads: list[dict[str, Any]] = []
    trace_headers: list[str | None] = []

    class SlowRequest:
        def __init__(self, timeout_s: float) -> None:
            self.timeout_s = timeout_s

        async def __aenter__(self) -> Any:
            await asyncio.sleep(self.timeout_s)
            raise asyncio.TimeoutError

        async def __aexit__(self, *_: object) -> None:
            return None

    class FakeSession:
        def __init__(self, *, timeout: Any) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        def post(
            self, url: str, *, json: dict[str, Any], headers: dict[str, str]
        ) -> SlowRequest:
            assert url == "http://opponent/evaluate"
            payloads.append(json)
            trace_headers.append(headers.get("X-Trace-ID"))
            return SlowRequest(float(self.timeout.total))

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)

    async def run() -> tuple[OpponentTimeout, float]:
        started = time.perf_counter()
        with pytest.raises(OpponentTimeout) as raised:
            await request_opponent(
                "http://opponent",
                proposal=_proposal(),
                graph_snapshot="toy-v1",
                samples=2,
                round_index=2,
                timeout_s=0.06,
                retries=4,
                trace_id="questioner:trace-1",
            )
        return raised.value, time.perf_counter() - started

    error, elapsed = asyncio.run(run())

    assert elapsed < 0.15
    assert error.stage == "client_request"
    assert error.attempts == 1
    assert error.trace_id == "questioner:trace-1"
    assert len(payloads) == 1
    assert payloads[0]["evaluation_timeout_s"] == pytest.approx(0.054, abs=0.001)
    assert payloads[0]["trace_id"] == "questioner:trace-1"
    assert trace_headers == ["questioner:trace-1"]


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [(422, ValueError), (500, RuntimeError)],
)
def test_request_opponent_does_not_hide_rejected_configuration_or_server_bug(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected_error: type[Exception],
) -> None:
    import aiohttp

    calls = 0

    class RejectedResponse(_Response):
        async def __aenter__(self) -> RejectedResponse:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class FakeSession:
        def __init__(self, *, timeout: Any) -> None:
            del timeout

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        def post(self, *_: object, **__: object) -> RejectedResponse:
            nonlocal calls
            calls += 1
            return RejectedResponse(
                '{"error":"ValueError","detail":"invalid relation catalog"}',
                status=status,
                content_type="application/json",
            )

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)

    async def run() -> None:
        with pytest.raises(expected_error, match=f"status {status}.*invalid relation catalog"):
            await request_opponent(
                "http://opponent",
                proposal=_proposal(),
                graph_snapshot="toy-v1",
                samples=2,
                round_index=2,
                timeout_s=1.0,
                retries=4,
            )

    asyncio.run(run())
    assert calls == 1


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [
        (422, ValueError),
        (500, RuntimeError),
        (502, OpponentUnavailable),
        (504, OpponentTimeout),
    ],
)
def test_request_opponent_classifies_non_json_http_failures(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected_error: type[Exception],
) -> None:
    import aiohttp

    class FailedResponse(_Response):
        async def __aenter__(self) -> FailedResponse:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class FakeSession:
        def __init__(self, *, timeout: Any) -> None:
            del timeout

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        def post(self, *_: object, **__: object) -> FailedResponse:
            return FailedResponse("gateway body", status=status)

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)

    async def run() -> None:
        with pytest.raises(expected_error):
            await request_opponent(
                "http://opponent",
                proposal=_proposal(),
                graph_snapshot="toy-v1",
                samples=2,
                round_index=2,
                timeout_s=1.0,
                retries=0,
            )

    asyncio.run(run())


def test_sglang_non_json_500_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _Response("overloaded", status=500),
            _Response(
                '{"choices":[{"message":{"role":"assistant","content":"ok"}}]}',
                status=200,
                content_type="application/json",
            ),
        ]
    )

    class ResponseContext:
        def __init__(self, response: _Response) -> None:
            self.response = response

        async def __aenter__(self) -> _Response:
            return self.response

        async def __aexit__(self, *_: object) -> None:
            return None

    class FakeRemoteSession:
        closed = False

        def __init__(self) -> None:
            self.calls = 0

        def post(self, *_: object, **__: object) -> ResponseContext:
            self.calls += 1
            return ResponseContext(next(responses))

        async def close(self) -> None:
            self.closed = True

    async def no_delay(_: float) -> None:
        return None

    service = FrozenSolverService(
        model_url="http://sglang",
        model="test-model",
        archive_path=tmp_path / "archive.sqlite",
        model_request_retries=1,
    )
    session = FakeRemoteSession()
    service._remote_session = session
    monkeypatch.setattr(opponent_module.asyncio, "sleep", no_delay)

    async def run() -> dict[str, Any]:
        try:
            return await service._remote_completion([], use_tools=False)
        finally:
            await service.close()

    assert asyncio.run(run())["content"] == "ok"
    assert session.calls == 2


class _ControlledEvaluationService(FrozenSolverService):
    def __init__(self, tmp_path: Path, *, delay_s: float, fail_sample: int | None = None) -> None:
        super().__init__(
            model_url="http://unused",
            model="controlled",
            archive_path=tmp_path / "archive.sqlite",
            candidate_archive_path=tmp_path / "candidates.sqlite",
            cache_evaluations=True,
        )
        self.delay_s = delay_s
        self.fail_sample = fail_sample
        self.started_samples: list[int] = []
        self.cancelled_samples: list[int] = []

    async def rollout(self, task: Any, backend: Any, **kwargs: Any) -> dict[str, float]:
        del task, backend
        sample_index = int(kwargs["sample_index"])
        self.started_samples.append(sample_index)
        try:
            if sample_index == self.fail_sample:
                raise OpponentUnavailable("sample failed")
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
            "program_operators": 1.0,
            "passage_searches": 0.0,
            "latency_ms": self.delay_s * 1000,
        }


def _evaluation_payload(*, timeout_s: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "proposal": _proposal().model_dump(mode="json"),
        "graph_snapshot": "toy-v1",
        "samples": 3,
        "round": 2,
        "interaction_mode": "graphscript",
        "graphscript_version": "0.1",
        "seed": 17,
        "trace_id": "questioner:trace-server",
    }
    if timeout_s is not None:
        payload["evaluation_timeout_s"] = timeout_s
    return payload


def test_server_deadline_cancels_siblings_clears_cache_and_skips_archive(
    tmp_path: Path,
) -> None:
    service = _ControlledEvaluationService(tmp_path, delay_s=0.2)

    async def run() -> tuple[OpponentTimeout, dict[str, Any]]:
        with pytest.raises(OpponentTimeout) as raised:
            await service.evaluate(_evaluation_payload(timeout_s=0.03))
        assert service._evaluation_cache == {}
        assert service.candidate_archive is not None
        assert service.candidate_archive.all() == []

        service.delay_s = 0.0
        recovered = await service.evaluate(_evaluation_payload(timeout_s=0.5))
        return raised.value, recovered

    try:
        error, recovered = asyncio.run(run())
        assert error.stage == "server_evaluation"
        assert error.attempts == 1
        assert error.trace_id == "questioner:trace-server"
        assert sorted(service.started_samples[:3]) == [0, 1, 2]
        assert sorted(service.cancelled_samples) == [0, 1, 2]
        assert recovered["samples"] == 3
        assert service.candidate_archive is not None
        assert len(service.candidate_archive.all()) == 1
    finally:
        asyncio.run(service.close())


def test_completed_cache_ignores_deadline_and_trace_control_metadata(
    tmp_path: Path,
) -> None:
    service = _ControlledEvaluationService(tmp_path, delay_s=0.0)
    first_payload = _evaluation_payload(timeout_s=0.5)
    second_payload = _evaluation_payload(timeout_s=0.2)
    second_payload["trace_id"] = "questioner:another-trace"

    async def run() -> tuple[dict[str, Any], dict[str, Any]]:
        first = await service.evaluate(first_payload)
        second = await service.evaluate(second_payload)
        return first, second

    try:
        first, second = asyncio.run(run())
        assert first == second
        assert sorted(service.started_samples) == [0, 1, 2]
        assert len(service._evaluation_cache) == 1
        assert service.candidate_archive is not None
        assert len(service.candidate_archive.all()) == 1
    finally:
        asyncio.run(service.close())


def test_shorter_cached_waiter_cancels_shared_work_without_raw_cancellation(
    tmp_path: Path,
) -> None:
    service = _ControlledEvaluationService(tmp_path, delay_s=0.2)
    long_payload = _evaluation_payload(timeout_s=0.5)
    short_payload = _evaluation_payload(timeout_s=0.03)
    short_payload["trace_id"] = "questioner:short-waiter"

    async def run() -> list[object]:
        long_waiter = asyncio.create_task(service.evaluate(long_payload))
        while len(service.started_samples) < 3:
            await asyncio.sleep(0)
        short_waiter = asyncio.create_task(service.evaluate(short_payload))
        return list(
            await asyncio.gather(long_waiter, short_waiter, return_exceptions=True)
        )

    try:
        outcomes = asyncio.run(run())
        assert all(isinstance(outcome, OpponentTimeout) for outcome in outcomes)
        assert sorted(service.started_samples) == [0, 1, 2]
        assert sorted(service.cancelled_samples) == [0, 1, 2]
        assert service._evaluation_cache == {}
        assert service.candidate_archive is not None
        assert service.candidate_archive.all() == []
    finally:
        asyncio.run(service.close())


def test_failed_sample_cancels_siblings_and_is_not_cached_or_archived(tmp_path: Path) -> None:
    service = _ControlledEvaluationService(tmp_path, delay_s=0.2, fail_sample=1)

    async def run() -> None:
        with pytest.raises(OpponentUnavailable, match="sample failed"):
            await service.evaluate(_evaluation_payload(timeout_s=0.5))
        assert service._evaluation_cache == {}
        assert service.candidate_archive is not None
        assert service.candidate_archive.all() == []

    try:
        asyncio.run(run())
        assert service.cancelled_samples
    finally:
        asyncio.run(service.close())


def test_service_does_not_misclassify_application_timeout_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ControlledEvaluationService(tmp_path, delay_s=0.0)

    async def broken_rollout(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        raise TimeoutError("rollout implementation bug")

    monkeypatch.setattr(service, "rollout", broken_rollout)

    async def run() -> None:
        with pytest.raises(TimeoutError, match="rollout implementation bug"):
            await service.evaluate(_evaluation_payload(timeout_s=0.5))
        assert service._evaluation_cache == {}
        assert service.candidate_archive is not None
        assert service.candidate_archive.all() == []

    try:
        asyncio.run(run())
    finally:
        asyncio.run(service.close())


def test_external_cancellation_is_not_converted_to_opponent_timeout(tmp_path: Path) -> None:
    service = _ControlledEvaluationService(tmp_path, delay_s=1.0)

    async def run() -> None:
        evaluation = asyncio.create_task(service.evaluate(_evaluation_payload()))
        while len(service.started_samples) < 3:
            await asyncio.sleep(0)
        evaluation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await evaluation
        assert service._evaluation_cache == {}
        assert service.candidate_archive is not None
        assert service.candidate_archive.all() == []

    try:
        asyncio.run(run())
        assert sorted(service.cancelled_samples) == [0, 1, 2]
    finally:
        asyncio.run(service.close())
