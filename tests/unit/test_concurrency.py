from threading import Barrier, Event, Thread

import pytest

from graphtask_r1.utils import ordered_parallel_map


def test_ordered_parallel_map_runs_concurrently_and_preserves_order() -> None:
    barrier = Barrier(2)

    def transform(value: int) -> int:
        if value < 2:
            barrier.wait(timeout=2)
        return value * 2

    assert list(ordered_parallel_map(transform, range(4), workers=2)) == [0, 2, 4, 6]


def test_ordered_parallel_map_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(ordered_parallel_map(str, [1], workers=0))


def test_ordered_parallel_map_keeps_workers_fed_behind_slow_first_record() -> None:
    release_first = Event()
    later_record_started = Event()
    results: list[int] = []

    def transform(value: int) -> int:
        if value == 0:
            release_first.wait(timeout=2)
        if value >= 4:
            later_record_started.set()
        return value

    thread = Thread(
        target=lambda: results.extend(ordered_parallel_map(transform, range(8), workers=2))
    )
    thread.start()
    try:
        assert later_record_started.wait(timeout=1)
    finally:
        release_first.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert results == list(range(8))
