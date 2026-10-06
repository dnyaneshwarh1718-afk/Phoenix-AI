# Phoenix AI — Phase 1 Quantitative Evaluation

Source suite: `Phoenix AI Real-World E2E Evaluation v3`
Source timestamp: `2026-10-05T19:17:03Z`

## Metrics

| Metric | Value |
|---|---:|
| Retrieval relevance (provenance) | 100.00% |
| Grounding validation pass rate | 100.00% |
| Citation completeness | 100.00% |
| Citation correctness | 100.00% |
| Expected-term coverage | 100.00% |
| Abstention / fail-closed accuracy | 100.00% |
| Mean confidence | 0.6782 |
| Mean latency | 2.194s |
| P50 latency | 2.988s |
| P95 latency | 4.005s |
| P99 latency | 4.313s |
| RAG mean latency | 3.392s |
| RAG P95 latency | 4.153s |
| Phase 1 composite score | 100.00% |

## Stage latency profile

| Stage | Count | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| agent.application | 2 | 0.002s | 0.002s | 0.004s | 0.004s | 0.004s |
| agent.memory | 4 | 0.009s | 0.008s | 0.019s | 0.020s | 0.020s |
| agent.rag | 9 | 3.364s | 3.289s | 4.133s | 4.323s | 4.371s |
| citation_build | 18 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| classification | 16 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| context_build | 18 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| document_prepare | 18 | 0.489s | 0.463s | 0.752s | 0.752s | 0.752s |
| document_total | 18 | 3.364s | 3.289s | 4.371s | 4.371s | 4.371s |
| generation | 18 | 2.051s | 1.937s | 2.929s | 2.929s | 2.929s |
| memory_retrieval | 16 | 0.017s | 0.015s | 0.027s | 0.027s | 0.027s |
| orchestration_total | 16 | 1.918s | 2.830s | 3.942s | 4.298s | 4.387s |
| planning | 16 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| planning_llm_calls | 1 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| planning_llm_total | 1 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| rag_total | 18 | 2.875s | 2.855s | 3.889s | 3.889s | 3.889s |
| retrieval | 18 | 0.824s | 0.917s | 0.989s | 0.989s | 0.989s |
| validation | 18 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |

Timing coverage: 100.0%

## Metric semantics

- **Retrieval relevance** is intentionally provenance-based. The current E2E result schema does not expose the full retrieved candidate list and relevance labels needed for true Precision@K/Recall@K.
- **Grounding validation** uses Phoenix's existing validator signal rather than an LLM-as-judge.
- **Citation completeness/correctness** are computed from the citations emitted by Phoenix and the expected evaluation document.
- **Latency** is computed from the E2E `latency_seconds` measurements.

## Upgrade path

When Phoenix starts recording the complete ranked candidate list plus gold chunk IDs, this evaluator can be extended with Precision@K, Recall@K, MRR, and nDCG without changing the report contract.
