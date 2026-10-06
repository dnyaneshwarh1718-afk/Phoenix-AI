# Phoenix AI Phase 2.1 — Closed-Loop Integration Validation

Phase 2.1 validates the new Orchestrator → Memory → Planning → Execution → Verification → Memory Update lifecycle.

## Live evaluation

Start Phoenix normally with Qdrant and Ollama available, then run:

```powershell
python -m evaluation.scripts.run_phase2_evaluation
```

Optional:

```powershell
python -m evaluation.scripts.run_phase2_evaluation --base-url http://127.0.0.1:8000 --user-id phase2-evaluator --project-id phoenix-phase2
```

The script writes `evaluation/results/phase2_latest.json`.

## Acceptance gates

- memory continuity = PASS
- closed-loop plan execution = PASS
- deterministic safety fail-closed = PASS
- downstream steps receive upstream verified results
- task outcomes are persisted only after deterministic verification

The existing Phase 1 regression suite remains mandatory. Phase 2.1 is additive and must not weaken the established 16/16, 100% composite baseline.

## Design changes in 2.1

1. Analytical planning fast path now covers natural analytical goals such as `Analyze the sales dataset...`, not only prompts containing the phrase `execution plan`.
2. `PlanExecutor` propagates verified upstream step results to downstream agent context through `metadata.plan_prior_results`.
3. General-agent execution explicitly receives current-plan prior results and is instructed not to invent missing outputs.
4. A live Phase 2 evaluator verifies memory continuity, closed-loop execution, safety, verification, and memory write-back.
