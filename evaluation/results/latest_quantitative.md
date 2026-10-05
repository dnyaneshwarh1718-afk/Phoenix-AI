# Phoenix AI — Phase 1 Quantitative Evaluation

Source suite: `Phoenix AI Real-World E2E Evaluation v3`
Source timestamp: `2026-10-05T18:43:42Z`

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
| Mean latency | 2.159s |
| P50 latency | 3.032s |
| P95 latency | 3.815s |
| P99 latency | 4.144s |
| RAG mean latency | 3.343s |
| RAG P95 latency | 3.973s |
| Phase 1 composite score | 100.00% |

## Stage latency profile

| Stage | Count | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| agent.application | 2 | 0.005s | 0.005s | 0.008s | 0.008s | 0.008s |
| agent.memory | 4 | 0.007s | 0.007s | 0.014s | 0.015s | 0.015s |
| agent.rag | 9 | 3.307s | 3.307s | 3.947s | 4.148s | 4.199s |
| citation_build | 18 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| classification | 16 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| context_build | 18 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| document_prepare | 18 | 0.477s | 0.430s | 0.692s | 0.692s | 0.692s |
| document_total | 18 | 3.307s | 3.307s | 4.198s | 4.198s | 4.198s |
| generation | 18 | 2.110s | 1.979s | 2.982s | 2.982s | 2.982s |
| memory_retrieval | 16 | 0.020s | 0.016s | 0.039s | 0.078s | 0.088s |
| orchestration_total | 16 | 1.887s | 2.743s | 3.743s | 4.127s | 4.223s |
| planning | 16 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| planning_llm_calls | 1 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| planning_llm_total | 1 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| rag_total | 18 | 2.830s | 2.781s | 3.768s | 3.768s | 3.768s |
| retrieval | 18 | 0.719s | 0.786s | 0.929s | 0.929s | 0.929s |
| validation | 18 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |

Timing coverage: 100.0%

## Metric semantics

- **Retrieval relevance** is intentionally provenance-based. The current E2E result schema does not expose the full retrieved candidate list and relevance labels needed for true Precision@K/Recall@K.
- **Grounding validation** uses Phoenix's existing validator signal rather than an LLM-as-judge.
- **Citation completeness/correctness** are computed from the citations emitted by Phoenix and the expected evaluation document.
- **Latency** is computed from the E2E `latency_seconds` measurements.

## Upgrade path

When Phoenix starts recording the complete ranked candidate list plus gold chunk IDs, this evaluator can be extended with Precision@K, Recall@K, MRR, and nDCG without changing the report contract.
