from evaluation.metrics import _stage_timing_stats


def test_stage_timing_stats_are_aggregated():
    results = [
        {"metadata": {"stage_timings": {"classification": 1.0, "memory_retrieval": 2.0}}},
        {"metadata": {"stage_timings": {"classification": 3.0}}},
        {"metadata": {"performance_timings": {"generation": 4.0}}},
    ]
    stats = _stage_timing_stats(results)
    assert stats["classification"]["count"] == 2
    assert stats["classification"]["mean_seconds"] == 2.0
    assert stats["memory_retrieval"]["max_seconds"] == 2.0
    assert stats["generation"]["p95_seconds"] == 4.0


def test_stage_timing_stats_ignore_invalid_values():
    results = [{"metadata": {"stage_timings": {"ok": 1.0, "bad": "1.0", "nan": float("nan")}}}]
    stats = _stage_timing_stats(results)
    assert set(stats) == {"ok"}
