# Phase 1 Validation Record

## Completed
- Added reproducible quantitative evaluator.
- Added ground-truth case schema.
- Added provenance-based retrieval relevance metric.
- Added grounding validation metric.
- Added citation completeness and correctness metrics.
- Added expected-term coverage.
- Added abstention/fail-closed accuracy.
- Added mean/P50/P95/P99 latency.
- Added composite score.
- Added independent evaluator tests.
- Added evaluation corpus fixtures for supported file formats.
- Added project dependency metadata.

## Validation performed in this build environment
- `pytest tests/test_quantitative_evaluation.py tests/test_real_world_eval_contract.py -q` → **8 passed**
- `python -m compileall -q app evaluation tests` → **passed**
- Quantitative evaluator executed successfully against the reconstructed validation fixture.
- Baseline report generated under `evaluation/results/latest_quantitative.*`.

## Baseline snapshot
The committed E2E suite is 14/14 passed. The quantitative report in this package is marked `source_type=reconstructed_validation_fixture` because the uploaded `app.zip` and `tests.zip` did not contain the full `evaluation/results/latest.json` payload. It was reconstructed from the committed E2E summary and visible result metadata solely to validate the Phase 1 evaluator.

**Do not treat the baseline quantitative values as a fresh live rerun.** To produce the authoritative live metrics, run the existing live E2E evaluation in the configured Phoenix environment and then run:

```powershell
python -m evaluation.scripts.run_quantitative_evaluation --results evaluation/results/latest.json
```

## Dependency boundary
The build environment used for this packaging task does not have network access, so missing Phoenix runtime dependencies such as `qdrant-client` and `langgraph` could not be installed. The full Phoenix runtime suite therefore was not falsely marked green. The Phase 1 evaluator itself is dependency-light and was fully validated independently.
