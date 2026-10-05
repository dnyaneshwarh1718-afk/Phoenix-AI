# Phoenix AI Evaluation

## Phase 1: quantitative evaluation

`metrics.py` converts a completed E2E result (`results/latest.json`) plus `datasets/cases.json` into a reproducible quantitative report.

### Metrics

1. Retrieval relevance — provenance-based expected-document hit rate for answerable RAG cases.
2. Grounding validation pass rate — Phoenix validator state matches answerability expectation.
3. Citation completeness — answerable RAG cases with at least one citation.
4. Citation correctness — citation points to the expected evaluation document.
5. Expected-term coverage — lexical ground-truth coverage in the answer.
6. Abstention/fail-closed accuracy — unanswerable and safety cases handled correctly.
7. Mean, P50, P95 and P99 latency.
8. Composite Phase 1 score.

### Run

```powershell
python -m evaluation.scripts.run_quantitative_evaluation --results evaluation/results/latest.json
```

The live E2E runner should be executed first in a configured Phoenix environment with FastAPI, Qdrant and Ollama available.

### Retrieval metric limitation

The current result schema does not persist the complete ranked retrieval candidate set or gold chunk IDs. Therefore Phase 1 intentionally does **not** invent Precision@K, Recall@K, MRR or nDCG. The retrieval metric is explicitly provenance-based. Once ranked candidates + gold labels are recorded, those metrics can be added without changing the report interface.
