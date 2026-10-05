# Phoenix AI — Phase 1.1 Latency Profiling

## Objective

Instrument Phoenix AI so the existing E2E evaluation can explain *where* request latency is spent instead of reporting only end-to-end latency.

## Instrumented stages

- `classification`
- `memory_retrieval`
- `planning`
- `agent.<name>` for specialized agent execution
- `orchestration_total`
- RAG internals via `metadata.performance_timings`:
  - `document_prepare`
  - `retrieval`
  - `context_build`
  - `generation`
  - `validation`
  - `citation_build`
  - `rag_total`
- Planning LLM details:
  - `planning_llm_total`
  - `planning_llm_calls`
  - `planning_llm_calls_seconds`

## Reporting

`evaluation.metrics` now aggregates every instrumented stage with:

- count
- mean
- P50
- P95
- P99
- max

The report also exposes:

- stage timing coverage
- the stage with the highest mean latency

## Live workflow

1. Start Phoenix, Qdrant and Ollama normally.
2. Run the existing live E2E evaluation so `evaluation/results/latest.json` contains `metadata.stage_timings` / `metadata.performance_timings`.
3. Run:

```powershell
python -m evaluation.scripts.run_quantitative_evaluation --results evaluation/results/latest.json
```

4. Inspect:

```text
evaluation/results/latest_quantitative.json
evaluation/results/latest_quantitative.md
```

## Important

The evaluator remains backward compatible. If an older `latest.json` has no stage telemetry, the existing overall metrics are still calculated and stage timing coverage will be reported as 0%.

## Engineering baseline

The user's current live Phase 1 baseline is:

- Retrieval relevance: 100.00%
- Grounding validation: 100.00%
- Citation completeness: 100.00%
- Citation correctness: 100.00%
- Mean latency: 11.869s
- P95 latency: 51.336s
- P99 latency: 105.052s
- Composite: 91.67%

These numbers are the pre-instrumentation baseline. They must be regenerated after instrumentation using the user's local Phoenix/Qdrant/Ollama environment before claiming a post-change performance improvement.

## Safety correction included in this build

The Application Control executor now fails closed for the currently unsupported `close` operation. It returns `status=blocked` and `executed=false` instead of reporting a generic unsupported result, so the E2E and quantitative evaluators have a deterministic safety signal.

## Local validation performed in the build environment

- Python compileall: passed
- Phase 1.1 metric/stage timing tests: 8 passed
- Application/security/planning regression tests from the supplied test suite: 14 passed
- Additional close-operation fail-closed regression: included and passed as part of the 8 Phase 1.1 tests

The live Phoenix/Qdrant/Ollama E2E suite must still be executed in the user's local runtime because this build environment does not contain the external Qdrant/BM25/LangGraph runtime dependencies and does not have package-network access.
