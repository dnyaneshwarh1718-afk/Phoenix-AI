"""Validate that a live E2E latest.json contains Phase 1.1 telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="evaluation/results/latest.json")
    args = parser.parse_args()
    path = Path(args.results)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", data if isinstance(data, list) else [])
    if not isinstance(results, list):
        raise SystemExit("Invalid evaluation result shape: expected results list")

    covered = 0
    missing = []
    for item in results:
        metadata = item.get("metadata") or {}
        if metadata.get("stage_timings") or metadata.get("performance_timings"):
            covered += 1
        else:
            missing.append(item.get("id", "unknown"))

    coverage = (covered / len(results) * 100) if results else 0.0
    print(f"stage_timing_coverage={coverage:.2f}% ({covered}/{len(results)})")
    if missing:
        print("missing telemetry:", ", ".join(missing))
    return 0 if covered == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
