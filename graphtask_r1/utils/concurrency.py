from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
        pending: dict[Future[_R], int] = {}
        ready: dict[int, _R] = {}
        submitted = 0
        next_result = 0

        def submit_one() -> bool:
            nonlocal submitted
            try:
                value = next(iterator)
            except StopIteration:
                return False
            pending[executor.submit(function, value)] = submitted
            submitted += 1
            return True

        for _ in range(workers * 2):
            if not submit_one():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index = pending.pop(future)
                ready[index] = future.result()
                submit_one()
            while next_result in ready:
                yield ready.pop(next_result)
                next_result += 1
