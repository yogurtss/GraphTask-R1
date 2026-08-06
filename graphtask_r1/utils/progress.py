from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic


@dataclass
class ProgressLogger:
    """Emit compact, rate-limited, machine-readable progress messages."""

    operation: str
    total: int | None = None
    interval_s: float = 5.0
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("graphtask_r1.progress")
    )
    clock: Callable[[], float] = monotonic
    _started_at: float | None = field(default=None, init=False)
    _last_logged_at: float | None = field(default=None, init=False)

    def start(self, **fields: object) -> None:
        now = self.clock()
        self._started_at = now
        self._last_logged_at = now
        self._emit("started", completed=0, elapsed_s=0.0, **fields)

    def update(self, completed: int, **fields: object) -> None:
        now = self.clock()
        if self._started_at is None:
            self._started_at = now
            self._last_logged_at = now
        interval_elapsed = (
            self._last_logged_at is None or now - self._last_logged_at >= self.interval_s
        )
        if not interval_elapsed:
            return
        self._last_logged_at = now
        self._emit(
            "progress",
            completed=completed,
            elapsed_s=now - self._started_at,
            **fields,
        )

    def finish(self, completed: int, **fields: object) -> None:
        now = self.clock()
        if self._started_at is None:
            self._started_at = now
        self._last_logged_at = now
        self._emit(
            "completed",
            completed=completed,
            elapsed_s=now - self._started_at,
            **fields,
        )

    def _emit(
        self,
        phase: str,
        *,
        completed: int,
        elapsed_s: float,
        **fields: object,
    ) -> None:
        values: dict[str, object] = {
            "operation": self.operation,
            "phase": phase,
            "completed": completed,
        }
        if self.total is not None:
            values["total"] = self.total
            values["percent"] = round(completed * 100 / self.total, 1) if self.total else 100.0
        values["elapsed_s"] = round(elapsed_s, 1)
        values.update(fields)
        message = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
            for key, value in values.items()
        )
        self.logger.info(message)
