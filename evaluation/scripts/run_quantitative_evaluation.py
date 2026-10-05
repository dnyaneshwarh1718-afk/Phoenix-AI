from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.metrics import load_and_evaluate, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Phoenix AI Phase 1 quantitative metrics.")
    parser.add_argument("--results", default="evaluation/results/latest.json")
    parser.add_argument("--cases", default="evaluation/datasets/cases.json")
    parser.add_argument("--output-json", default="evaluation/results/latest_quantitative.json")
    parser.add_argument("--output-md", default="evaluation/results/latest_quantitative.md")
    args = parser.parse_args()

    metrics = load_and_evaluate(args.results, args.cases)
    write_report(metrics, args.output_json, args.output_md)

    m = metrics["measured_metrics"]
    print("Phoenix AI Phase 1 quantitative evaluation")
    print(f"retrieval_relevance={m['retrieval_relevance_rate_percent']:.2f}%")
    print(f"grounding_validation={m['grounding_validation_pass_rate_percent']:.2f}%")
    print(f"citation_completeness={m['citation_completeness_percent']:.2f}%")
    print(f"citation_correctness={m['citation_correctness_percent']:.2f}%")
    print(f"mean_latency={m['latency_seconds']['mean_seconds']:.3f}s")
    print(f"p95_latency={m['latency_seconds']['p95_seconds']:.3f}s")
    print(f"p99_latency={m['latency_seconds']['p99_seconds']:.3f}s")
    print(f"composite={m['phase1_quality_score_percent']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
