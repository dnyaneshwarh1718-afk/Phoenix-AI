# Phase 1.1 Corrective Build

This build addresses the live evaluation findings:

- planning was ~115 s on the local Qwen3 planner;
- XLSX analytical retrieval returned an unverified answer;
- agent-local performance timings were not normalized into the canonical stage timing bucket;
- live evaluation results need a contract check so missing telemetry cannot be mistaken for valid profiling.

## Changes

1. Qwen3 planning requests set `think=false` and use a bounded `num_predict=768`.
2. Agent `performance_timings` are copied into `metadata.stage_timings`.
3. Excel ingestion preserves each row's product/value relationship in semantic text.
4. Added a forced evaluation-corpus reindex script.
5. Added a live-result telemetry contract validator.

## Validation sequence

```powershell
python -m evaluation.scripts.reindex_evaluation_corpus
python -m evaluation.scripts.run_live_evaluation
python -m evaluation.scripts.validate_live_result_contract --results evaluation/results/latest.json
python -m evaluation.scripts.run_quantitative_evaluation --results evaluation/results/latest.json
```

The reindex command is required once after the Excel ingestion change so Qdrant/BM25 do not serve the previous chunk representation.
