# Phoenix AI — Phase 1 Quantitative Evaluation

Source suite: `Phoenix AI Real-World E2E Evaluation v3`
Source timestamp: `2026-10-04T17:58:31Z`

## Metrics

| Metric | Value |
|---|---:|
| Retrieval relevance (provenance) | 100.00% |
| Grounding validation pass rate | 100.00% |
| Citation completeness | 100.00% |
| Citation correctness | 100.00% |
| Expected-term coverage | 88.89% |
| Abstention / fail-closed accuracy | 100.00% |
| Mean confidence | 0.6382 |
| Mean latency | 11.869s |
| P50 latency | 3.779s |
| P95 latency | 51.336s |
| P99 latency | 105.052s |
| RAG mean latency | 5.277s |
| RAG P95 latency | 11.083s |
| Phase 1 composite score | 98.15% |

## Metric semantics

- **Retrieval relevance** is intentionally provenance-based. The current E2E result schema does not expose the full retrieved candidate list and relevance labels needed for true Precision@K/Recall@K.
- **Grounding validation** uses Phoenix's existing validator signal rather than an LLM-as-judge.
- **Citation completeness/correctness** are computed from the citations emitted by Phoenix and the expected evaluation document.
- **Latency** is computed from the E2E `latency_seconds` measurements.

## Upgrade path

When Phoenix starts recording the complete ranked candidate list plus gold chunk IDs, this evaluator can be extended with Precision@K, Recall@K, MRR, and nDCG without changing the report contract.
