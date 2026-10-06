# Phoenix AI Phase 2 Validation

## Phase 2 closed-loop contract

The graph now follows:

`classify → memory retrieval → route → plan-if-needed → execute → verify → memory update → response`

### Regression gate

Run the existing Phase 1 suite first:

```powershell
python -m evaluation.scripts.run_live_evaluation
python -m evaluation.scripts.validate_live_result_contract --results evaluation/results/latest.json
python -m evaluation.scripts.run_quantitative_evaluation --results evaluation/results/latest.json
```

Required Phase 1 gate:

- 16/16 E2E cases pass
- stage timing coverage = 100%
- retrieval relevance = 100%
- grounding validation = 100%
- citation completeness = 100%
- citation correctness = 100%

### Phase 2 scenarios

1. **Memory continuity**
   - Store a user fact.
   - Ask a semantically related question in a later request.
   - Confirm `memory_retrieved > 0` and the response uses only relevant memory context.

2. **Simple direct routing**
   - RAG, research, application, and memory requests continue to bypass unnecessary planning.

3. **Closed-loop planning**
   - A complex non-planning request creates a validated plan.
   - The plan executor dispatches dependent steps to specialized agents.
   - A failed/unverified step stops dependent execution.

4. **Safety**
   - A blocked application operation remains `executed=false` and is verified as a safe block.

5. **Memory update**
   - Successful multi-step tasks create a compact `task_outcome` memory.
   - Explicit MemoryAgent writes are not duplicated as task outcomes.

## Local validation performed during build

- Python `compileall` passes.
- Plan executor dependency-order smoke test passes.
- Plan executor fail-closed smoke test passes.
- Safe application-block smoke test passes.
- Phase 2 graph import/verification smoke test passes with the graph dependency boundary isolated.

The full live E2E gate must be executed in the user's Windows `.venv` because authoritative Phoenix runtime dependencies and local Qdrant/Ollama services are not available in the build environment.
