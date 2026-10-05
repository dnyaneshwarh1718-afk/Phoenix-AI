# Phoenix AI Phase 1.1 — Iteration 2

## Live findings addressed

The latest live evaluation showed:

- `RAG-XLSX-02` remained unverified because the updated Excel loader had not yet been persisted to the existing Qdrant/BM25 indexes.
- `PLAN-01` remained the dominant latency contributor (~131.6 s).
- Stage timing coverage reached 100%, so the telemetry path is now working.

## Changes

### 1. Deterministic planning fast path

Phoenix now compiles a conservative, low-risk data-analysis execution plan without invoking the local LLM when the request explicitly asks for an execution plan and clearly describes a dataset/revenue/top-products workflow.

This is a general planner optimization: ambiguous or novel planning requests continue through the validated LLM planner.

### 2. Bounded LLM planning

For requests that require the LLM planner:

- maximum attempts: 2
- maximum generated tokens: 384
- Ollama reasoning (`think`) remains disabled for `task_type=reasoning`
- JSON mode remains enabled for structured plans

These defaults can be overridden through settings/environment variables.

### 3. Excel row semantics

The Excel loader preserves header + row relationships. After changing the loader, the persistent indexes must be rebuilt before evaluation.

## Required live validation

Run from the Phoenix project root:

```powershell
python -m evaluation.scripts.reindex_evaluation_corpus
python -m evaluation.scripts.run_live_evaluation
python -m evaluation.scripts.validate_live_result_contract --results evaluation/results/latest.json
python -m evaluation.scripts.run_quantitative_evaluation --results evaluation/results/latest.json
```

Do not skip reindexing. Existing Qdrant/BM25 data can otherwise continue serving the previous Excel chunk representation.

## Success criteria

- E2E: `16/16`
- RAG grounding: `100%`
- citation completeness: `100%`
- citation correctness: `100%`
- stage timing coverage: `100%`
- fail-closed security scenario: pass
- planning latency: materially lower than the previous ~131.6 s baseline
