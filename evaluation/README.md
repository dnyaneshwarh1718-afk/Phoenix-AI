# Phoenix AI — Real-World End-to-End Evaluation v3

This is a live acceptance harness for a **running Phoenix AI instance**. It does not replace pytest. It exercises the real HTTP API, real local Ollama model, Qdrant retrieval, real PDF/DOCX/XLSX/PPTX/TXT files, Planning, Memory, Application Control, routing and safety.

## Start the stack

From `R:\Phoenix-AI-main`:

```powershell
.\.venv\Scripts\Activate.ps1
```

Terminal 1:

```powershell
# Qdrant should already be running
Invoke-RestMethod http://localhost:6333
```

Terminal 2:

```powershell
ollama list
```

Ensure `qwen3:4b-instruct` is present.

Terminal 3:

```powershell
uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Place the evaluation folder

Extract this ZIP so that the following exists under the Phoenix project root:

```text
evaluation/
  corpus/
  datasets/
  scripts/
```

Therefore the application test can use:

```text
evaluation/corpus/phoenix_evaluation.xlsx
```

## Install evaluator dependency

The evaluator only needs `httpx`:

```powershell
python -m pip install httpx
```

The PDF fixture is already included. You do **not** need ReportLab to run the evaluation.

## Preflight

```powershell
python evaluation/scripts/run_live_evaluation.py --preflight
```

Expected:

```text
Corpus: OK
Phoenix: 200
Qdrant: 200
Ollama: 200
Model: qwen3:4b-instruct FOUND
```

## Run the complete live E2E evaluation

```powershell
python evaluation/scripts/run_live_evaluation.py
```

The runner performs:

1. TXT RAG retrieval
2. DOCX RAG retrieval
3. PDF RAG retrieval
4. PPTX RAG retrieval
5. XLSX retrieval and data reasoning
6. hallucination resistance
7. Planning
8. real Application Control inspection of the evaluation XLSX
9. close-operation safety boundary
10. document-precedence routing
11. two-turn persistent Memory recall

Results:

```text
evaluation/results/latest.json
evaluation/results/latest.md
```

Exit codes:

- `0`: all acceptance cases passed
- `1`: Phoenix ran but one or more acceptance cases failed
- `2`: environment/preflight failure

## Important

The runner uses user ID `e2e-evaluator` so benchmark memory is isolated from your normal user memory. Application Control's `APP-01` performs a read-only Excel inspection; it does not modify the workbook. `SEC-01` verifies that unsupported close behavior does not execute.

Do not judge the system from one answer. The generated report contains route, latency, response, metadata, evidence count and failure reason for every live case.
