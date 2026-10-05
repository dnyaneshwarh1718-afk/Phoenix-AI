from __future__ import annotations
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator


def _bucket(metadata: dict[str, Any]) -> dict[str, float]:
    timings = metadata.get("stage_timings")
    if not isinstance(timings, dict):
        timings = {}
        metadata["stage_timings"] = timings
    return timings


def add_timing(metadata: dict[str, Any], stage: str, seconds: float) -> None:
    timings = _bucket(metadata)
    timings[stage] = round(float(seconds), 6)


@contextmanager
def timed(metadata: dict[str, Any], stage: str) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        add_timing(metadata, stage, perf_counter() - started)
