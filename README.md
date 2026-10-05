# Phoenix AI — Phase 1 Quantitative Evaluation

Phoenix AI is a local-first modular agentic AI platform. This package contains the current application foundation plus the Phase 1 evaluation layer.

## Phase 1 goal

Turn the existing real-world E2E evaluation into reproducible quantitative metrics:

- Retrieval relevance (provenance-based)
- Grounding validation pass rate
- Citation completeness
- Citation correctness
- Expected-term coverage
- Abstention / fail-closed accuracy
- Mean / P50 / P95 / P99 latency
- Composite Phase 1 score

## Run

Install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the evaluator against a completed live E2E result:

```powershell
python -m evaluation.scripts.run_quantitative_evaluation `
  --results evaluation/results/latest.json
```

Outputs:

- `evaluation/results/latest_quantitative.json`
- `evaluation/results/latest_quantitative.md`

## Important metric boundary

The current E2E result schema contains citations but not the complete ranked candidate list with gold chunk labels. Therefore this phase reports **provenance-based retrieval relevance**, not Precision@K/Recall@K/MRR/nDCG. Those metrics are the next instrumentation upgrade once ranked candidate lists and gold chunk IDs are persisted.

## Validation

The quantitative evaluator is dependency-light and can be tested independently:

```powershell
pytest tests/test_quantitative_evaluation.py -q
```

The package also retains the existing Phoenix application and test suite. Full runtime validation requires the configured Python dependencies, Ollama, and Qdrant environment used by Phoenix.
