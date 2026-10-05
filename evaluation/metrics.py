from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | None
    numerator: int | float | None
    denominator: int | float | None
    definition: str


def _pct(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _normalized_text(value: Any) -> str:
    text = str(value or "").lower()
    # Make lexical matching robust to currency separators and punctuation:
    # "$14,400" should match the ground-truth token "14400".
    return "".join(ch for ch in text if ch.isalnum() or ch in "._:-").replace(",", "")


def _basename(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "/").rstrip("/").split("/")[-1].lower()


def _citation_list(result: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = result.get("metadata") or {}
    citations = metadata.get("citations") or []
    return [c for c in citations if isinstance(c, dict)]


def _citation_matches_expected(citation: dict[str, Any], expected_doc: str) -> bool:
    expected = _basename(expected_doc)
    candidates = [
        citation.get("file_name"),
        citation.get("source_path"),
        citation.get("document_reference"),
    ]
    return any(_basename(candidate) == expected for candidate in candidates if candidate)


def _result_map(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("id")): r for r in results if r.get("id")}


def _stage_timing_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, list[float]] = {}
    for result in results:
        metadata = result.get("metadata") or {}
        timings = metadata.get("performance_timings") or {}
        if not isinstance(timings, dict):
            continue
        for stage, value in timings.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.setdefault(str(stage), []).append(float(value))
        # Orchestration graph timings are already stored in metadata.stage_timings.
        graph_timings = metadata.get("stage_timings") or {}
        if isinstance(graph_timings, dict):
            for stage, value in graph_timings.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.setdefault(str(stage), []).append(float(value))

    return {
        stage: {
            "count": len(items),
            "mean_seconds": statistics.fmean(items),
            "p50_seconds": _percentile(items, 0.50),
            "p95_seconds": _percentile(items, 0.95),
            "p99_seconds": _percentile(items, 0.99),
            "max_seconds": max(items),
        }
        for stage, items in sorted(values.items())
    }


def evaluate(results_payload: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a completed E2E result JSON without requiring Phoenix runtime dependencies.

    The evaluator deliberately separates *measured* signals from assumptions:
    - Retrieval relevance is provenance-based: at least one returned citation points
      to the expected evaluation document. It is not a semantic relevance score.
    - Grounding pass rate uses Phoenix's own `metadata.valid` signal against the
      case's answerable/unanswerable expectation.
    - Citation completeness means an answerable RAG case contains >=1 citation.
    - Citation correctness means at least one citation points to the expected document.
    - Answer term accuracy is lexical coverage of case expected_terms.
    """
    results = results_payload.get("results") or []
    by_id = _result_map(results)
    rag_cases = [c for c in cases if c.get("category") == "rag"]
    answerable_rag = [c for c in rag_cases if not c.get("unanswerable")]
    unanswerable_rag = [c for c in rag_cases if c.get("unanswerable")]

    retrieval_hits = 0
    citation_complete = 0
    citation_correct = 0
    answer_term_hits = 0
    answer_term_total = 0
    grounding_correct = 0
    abstention_correct = 0
    confidence_values: list[float] = []
    rag_latency: list[float] = []
    all_latency: list[float] = []

    case_metrics: list[dict[str, Any]] = []

    for case in cases:
        cid = case["id"]
        result = by_id.get(cid)
        if result is None:
            case_metrics.append({"id": cid, "present": False, "passed": False, "reason": "missing result"})
            continue

        metadata = result.get("metadata") or {}
        response = str(result.get("response") or "")
        response_l = response.lower()
        response_normalized = _normalized_text(response)
        citations = _citation_list(result)
        latency = result.get("latency_seconds")
        if isinstance(latency, (int, float)):
            all_latency.append(float(latency))
            if case.get("category") == "rag":
                rag_latency.append(float(latency))

        confidence = metadata.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_values.append(float(confidence))

        expected_terms = [str(t).lower() for t in case.get("expected_terms", [])]
        matched_terms = [
            t for t in expected_terms
            if t in response_l or _normalized_text(t) in response_normalized
        ]
        forbidden_terms = [str(t).lower() for t in case.get("forbidden_terms", [])]
        forbidden_present = any(t in response_l for t in forbidden_terms)

        if case.get("category") == "rag":
            expected_doc = case.get("document_reference")
            doc_match = bool(expected_doc) and any(
                _citation_matches_expected(c, expected_doc) for c in citations
            )
            if not case.get("unanswerable") and doc_match:
                retrieval_hits += 1
                citation_correct += 1
            if not case.get("unanswerable") and citations:
                citation_complete += 1

            if expected_terms:
                answer_term_hits += len(matched_terms)
                answer_term_total += len(expected_terms)

            expected_valid = not bool(case.get("unanswerable"))
            actual_valid = bool(metadata.get("valid"))
            if actual_valid == expected_valid:
                grounding_correct += 1

            case_metrics.append({
                "id": cid,
                "category": "rag",
                "retrieval_provenance_match": doc_match,
                "citation_count": len(citations),
                "citation_complete": bool(citations) if not case.get("unanswerable") else None,
                "citation_correct": doc_match,
                "expected_terms": expected_terms,
                "matched_terms": matched_terms,
                "grounding_expected_valid": expected_valid,
                "grounding_actual_valid": actual_valid,
                "confidence": confidence,
                "latency_seconds": latency,
            })
        elif case.get("category") == "planning":
            plan_ok = bool(result.get("passed")) and (
                "execution plan" in response_l or "step_" in response_l
            )
            case_metrics.append({"id": cid, "category": "planning", "plan_structure_valid": plan_ok, "latency_seconds": latency})
        elif case.get("category") == "application":
            if case.get("safety") == "fail_closed":
                safety_ok = bool(result.get("passed")) and any(
                    token in response_l for token in ("approval", "blocked", "requires approval", "denied")
                )
                if safety_ok:
                    abstention_correct += 1
                case_metrics.append({"id": cid, "category": "application", "fail_closed_correct": safety_ok, "latency_seconds": latency})
            else:
                term_ok = all(t in response_l for t in expected_terms)
                execution_ok = bool(result.get("passed")) and (metadata.get("status") not in {"failed", "error"})
                case_metrics.append({"id": cid, "category": "application", "expected_terms_present": term_ok, "execution_signal_ok": execution_ok, "latency_seconds": latency})
        elif case.get("category") == "memory":
            case_metrics.append({"id": cid, "category": "memory", "passed": bool(result.get("passed")), "latency_seconds": latency})
        else:
            case_metrics.append({"id": cid, "category": case.get("category"), "passed": bool(result.get("passed")), "latency_seconds": latency})

    retrieval_rate = _pct(retrieval_hits, len(answerable_rag) if answerable_rag else 0)
    grounding_rate = _pct(grounding_correct, len(rag_cases))
    citation_complete_rate = _pct(citation_complete, len(answerable_rag))
    citation_correct_rate = _pct(citation_correct, len(answerable_rag))
    answer_term_rate = _pct(answer_term_hits, answer_term_total)

    for case in unanswerable_rag:
        result = by_id.get(case["id"])
        if result:
            response_l = str(result.get("response") or "").lower()
            metadata = result.get("metadata") or {}
            expected_valid = False
            actual_valid = bool(metadata.get("valid"))
            forbidden_terms = [str(t).lower() for t in case.get("forbidden_terms", [])]
            correct = (actual_valid is expected_valid) and not any(
                t in response_l or _normalized_text(t) in response_normalized
                for t in forbidden_terms
            )
            if correct:
                abstention_correct += 1

    abstention_rate = _pct(abstention_correct, len(unanswerable_rag) + sum(1 for c in cases if c.get("safety") == "fail_closed"))

    lat = {
        "count": len(all_latency),
        "mean_seconds": statistics.fmean(all_latency) if all_latency else None,
        "median_seconds": statistics.median(all_latency) if all_latency else None,
        "p50_seconds": _percentile(all_latency, 0.50),
        "p95_seconds": _percentile(all_latency, 0.95),
        "p99_seconds": _percentile(all_latency, 0.99),
        "rag_mean_seconds": statistics.fmean(rag_latency) if rag_latency else None,
        "rag_p95_seconds": _percentile(rag_latency, 0.95),
    }

    component_values = [
        v for v in [retrieval_rate, grounding_rate, citation_complete_rate,
                    citation_correct_rate, answer_term_rate, abstention_rate]
        if v is not None
    ]
    quality_score = statistics.fmean(component_values) if component_values else None

    stage_timings = _stage_timing_stats(results)

    passed = results_payload.get("passed")
    total = results_payload.get("total")
    return {
        "schema_version": "1.0",
        "source_type": results_payload.get("_fixture_note") and "reconstructed_validation_fixture" or "live_e2e_result",
        "source_suite": results_payload.get("suite"),
        "source_timestamp": results_payload.get("timestamp"),
        "source_pass_rate": results_payload.get("pass_rate"),
        "measured_metrics": {
            "retrieval_relevance_rate_percent": retrieval_rate,
            "retrieval_relevance_definition": "Share of answerable RAG cases with at least one citation whose file/source basename matches the expected evaluation document.",
            "grounding_validation_pass_rate_percent": grounding_rate,
            "grounding_definition": "Share of RAG cases where Phoenix metadata.valid matches the case answerability expectation (true for answerable, false for unanswerable).",
            "citation_completeness_percent": citation_complete_rate,
            "citation_completeness_definition": "Share of answerable RAG cases containing at least one citation.",
            "citation_correctness_percent": citation_correct_rate,
            "citation_correctness_definition": "Share of answerable RAG cases with at least one citation pointing to the expected evaluation document.",
            "answer_expected_term_coverage_percent": answer_term_rate,
            "abstention_and_fail_closed_accuracy_percent": abstention_rate,
            "confidence_mean": statistics.fmean(confidence_values) if confidence_values else None,
            "latency_seconds": lat,
            "stage_timings_seconds": stage_timings,
            "stage_timing_coverage_percent": _pct(sum(1 for r in results if (r.get("metadata") or {}).get("stage_timings") or (r.get("metadata") or {}).get("performance_timings")), len(results)),
            "stage_timing_bottleneck": max(
                stage_timings.items(),
                key=lambda item: item[1]["mean_seconds"],
                default=None,
            ),
            "phase1_quality_score_percent": quality_score,
        },
        "suite_counts": {
            "results_present": len(results),
            "passed": passed,
            "total": total,
            "answerable_rag_cases": len(answerable_rag),
            "unanswerable_rag_cases": len(unanswerable_rag),
        },
        "case_metrics": case_metrics,
    }


def load_and_evaluate(results_path: str | Path, cases_path: str | Path) -> dict[str, Any]:
    results_payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    return evaluate(results_payload, cases)


def write_report(metrics: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    Path(json_path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    m = metrics["measured_metrics"]
    lat = m["latency_seconds"]
    lines = [
        "# Phoenix AI — Phase 1 Quantitative Evaluation",
        "",
        f"Source suite: `{metrics.get('source_suite')}`",
        f"Source timestamp: `{metrics.get('source_timestamp')}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Retrieval relevance (provenance) | {m['retrieval_relevance_rate_percent']:.2f}% |",
        f"| Grounding validation pass rate | {m['grounding_validation_pass_rate_percent']:.2f}% |",
        f"| Citation completeness | {m['citation_completeness_percent']:.2f}% |",
        f"| Citation correctness | {m['citation_correctness_percent']:.2f}% |",
        f"| Expected-term coverage | {m['answer_expected_term_coverage_percent']:.2f}% |",
        f"| Abstention / fail-closed accuracy | {m['abstention_and_fail_closed_accuracy_percent']:.2f}% |",
        f"| Mean confidence | {m['confidence_mean']:.4f} |" if m["confidence_mean"] is not None else "| Mean confidence | N/A |",
        f"| Mean latency | {lat['mean_seconds']:.3f}s |",
        f"| P50 latency | {lat['p50_seconds']:.3f}s |",
        f"| P95 latency | {lat['p95_seconds']:.3f}s |",
        f"| P99 latency | {lat['p99_seconds']:.3f}s |",
        f"| RAG mean latency | {lat['rag_mean_seconds']:.3f}s |",
        f"| RAG P95 latency | {lat['rag_p95_seconds']:.3f}s |",
        f"| Phase 1 composite score | {m['phase1_quality_score_percent']:.2f}% |",
        "",
        "## Stage latency profile",
        "",
        "| Stage | Count | Mean | P50 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, stats in sorted(m.get("stage_timings_seconds", {}).items()):
        lines.append(
            f"| {stage} | {stats['count']} | {stats['mean_seconds']:.3f}s | {stats['p50_seconds']:.3f}s | {stats['p95_seconds']:.3f}s | {stats['p99_seconds']:.3f}s | {stats['max_seconds']:.3f}s |"
        )
    lines += [
        "",
        f"Timing coverage: {m.get('stage_timing_coverage_percent') if m.get('stage_timing_coverage_percent') is not None else 'N/A'}%",
        "",
        "## Metric semantics",
        "",
        "- **Retrieval relevance** is intentionally provenance-based. The current E2E result schema does not expose the full retrieved candidate list and relevance labels needed for true Precision@K/Recall@K.",
        "- **Grounding validation** uses Phoenix's existing validator signal rather than an LLM-as-judge.",
        "- **Citation completeness/correctness** are computed from the citations emitted by Phoenix and the expected evaluation document.",
        "- **Latency** is computed from the E2E `latency_seconds` measurements.",
        "",
        "## Upgrade path",
        "",
        "When Phoenix starts recording the complete ranked candidate list plus gold chunk IDs, this evaluator can be extended with Precision@K, Recall@K, MRR, and nDCG without changing the report contract.",
    ]
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
