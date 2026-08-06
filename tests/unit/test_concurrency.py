from threading import Barrier

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
