from __future__ import annotations

import logging

import pytest

from graphtask_r1.utils import ProgressLogger


def test_progress_logger_emits_start_periodic_and_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.progress")
    caplog.set_level(logging.INFO, logger="test.progress")
    ticks = iter((0.0, 1.0, 6.0, 8.0))
    progress = ProgressLogger(
        "data.test",
        total=4,
        interval_s=5.0,
        logger=logger,
        clock=lambda: next(ticks),
    )

    progress.start(source="fixture")
    progress.update(1)
    progress.update(2, accepted=2)
    progress.finish(4, accepted=4)

    messages = [record.message for record in caplog.records]
    assert len(messages) == 3
    assert 'operation="data.test" phase="started" completed=0 total=4' in messages[0]
    assert 'phase="progress" completed=2 total=4 percent=50.0' in messages[1]
    assert 'phase="completed" completed=4 total=4 percent=100.0' in messages[2]
