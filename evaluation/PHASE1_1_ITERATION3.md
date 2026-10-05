# Phoenix AI Phase 1.1 — Iteration 3

## Purpose

Resolve the final Phase 1.1 E2E failure for analytical spreadsheet questions without degrading the already validated latency, grounding, citation, memory, application-control, or safety paths.

## Live baseline before this iteration

- E2E: 15/16
- RAG-XLSX-02: failed with `Engine A`, `14400`, and `unverified`
- Stage timing coverage: 100%
- Mean latency: 2.293s
- P95 latency: 4.019s
- P99 latency: 4.299s
- Retrieval relevance: 100%
- Grounding validation: 88.89%
- Citation completeness: 100%
- Citation correctness: 100%

## Root cause

Hybrid dense/BM25 retrieval is designed to find relevant evidence, not execute spreadsheet aggregations. Reindexing preserved spreadsheet rows but did not change the fundamental limitation: a query such as "which product has the highest revenue?" requires comparing numeric values across rows.

## Solution

Added `app.rag.structured.ExcelAnalyticalRetriever`.

The structured retriever is read-only and activates only for an explicitly scoped Excel workbook and a supported analytical operation:

- highest / maximum / largest
- lowest / minimum / smallest
- total / sum
- average / mean
- top N

It reads the workbook with pandas, identifies the requested numeric field, computes the result deterministically, and returns the computed row/value as citation-ready evidence. Ordinary spreadsheet questions continue through dense + BM25 + RRF retrieval.

This avoids hardcoding the evaluation answer and establishes a reusable architecture for spreadsheet analytics.

## Validation

Local targeted tests: **13 passed**.

The live E2E test must be run in the user's Windows environment because it depends on the real Qdrant/Ollama services and the project's runtime installation.

## Completion gate

After deployment, run:

```powershell
python -m evaluation.scripts.run_live_evaluation
python -m evaluation.scripts.validate_live_result_contract --results evaluation/results/latest.json
python -m evaluation.scripts.run_quantitative_evaluation --results evaluation/results/latest.json
```

Expected Phase 1.1 gate:

- E2E = 16/16
- Stage timing coverage = 100%
- Grounding validation = 100%
- Citation completeness = 100%
- Citation correctness = 100%
- SEC-01 = PASS
- Latency remains near the improved ~2–4 second baseline rather than returning to the previous 100+ second planning tail
