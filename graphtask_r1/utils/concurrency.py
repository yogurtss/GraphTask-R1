from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def validate_workers(workers: int) -> int:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    return workers


def ordered_parallel_map(
    function: Callable[[_T], _R], values: Iterable[_T], *, workers: int
) -> Iterator[_R]:
    """Map records concurrently while yielding results in input order."""
    validate_workers(workers)
    if workers == 1:
        for value in values:
            yield function(value)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="graphtask-data") as executor:
        iterator = iter(values)
        pending: deque[Future[_R]] = deque()
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            with suppress(StopIteration):
                pending.append(executor.submit(function, next(iterator)))
