from evaluation.metrics import evaluate

def _fixture():
    import json
    from pathlib import Path
    return json.loads(Path("evaluation/results/fixture_latest.json").read_text()), json.loads(Path("evaluation/datasets/cases.json").read_text())

def test_all_cases_are_scored():
    payload, cases = _fixture()
    report = evaluate(payload, cases)
    assert len(report["case_metrics"]) == 14
    assert report["suite_counts"]["results_present"] == 14

def test_retrieval_grounding_and_citations():
    payload, cases = _fixture()
    m = evaluate(payload, cases)["measured_metrics"]
    assert m["retrieval_relevance_rate_percent"] == 100.0
    assert m["grounding_validation_pass_rate_percent"] == 100.0
    assert m["citation_completeness_percent"] == 100.0
    assert m["citation_correctness_percent"] == 100.0

def test_latency_percentiles_are_monotonic():
    payload, cases = _fixture()
    lat = evaluate(payload, cases)["measured_metrics"]["latency_seconds"]
    assert lat["mean_seconds"] > 0
    assert lat["p50_seconds"] <= lat["p95_seconds"] <= lat["p99_seconds"]

def test_abstention_is_counted():
    payload, cases = _fixture()
    m = evaluate(payload, cases)["measured_metrics"]
    assert m["abstention_and_fail_closed_accuracy_percent"] == 100.0

def test_missing_result_is_not_silent():
    payload, cases = _fixture()
    payload["results"] = payload["results"][:-1]
    report = evaluate(payload, cases)
    missing = [x for x in report["case_metrics"] if x.get("present") is False]
    assert len(missing) == 1
