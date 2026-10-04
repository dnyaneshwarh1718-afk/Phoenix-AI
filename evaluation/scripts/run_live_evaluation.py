from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation"
CORPUS = EVAL / "corpus"
DATA = EVAL / "datasets" / "cases.json"
RESULTS = EVAL / "results"
API = os.getenv("PHOENIX_API_URL", "http://127.0.0.1:8000")
QDRANT = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
USER = os.getenv("PHOENIX_EVAL_USER", "e2e-evaluator")

REQUIRED_FILES = [
    "phoenix_architecture.txt", "phoenix_security.txt", "phoenix_projects.txt",
    "phoenix_evaluation.pdf", "phoenix_evaluation.docx",
    "phoenix_evaluation.xlsx", "phoenix_evaluation.pptx",
]

def corpus_preflight() -> list[str]:
    return [x for x in REQUIRED_FILES if not (CORPUS / x).is_file() or (CORPUS / x).stat().st_size == 0]

def preflight() -> bool:
    missing = corpus_preflight()
    print("Corpus:", "OK" if not missing else f"FAIL: {missing}")
    if missing:
        return False
    try:
        with httpx.Client(timeout=10) as c:
            h = c.get(f"{API}/health"); print("Phoenix:", h.status_code); h.raise_for_status()
            q = c.get(f"{QDRANT}/collections"); print("Qdrant:", q.status_code); q.raise_for_status()
            o = c.get(f"{OLLAMA}/api/tags"); print("Ollama:", o.status_code); o.raise_for_status()
            names = [m.get("name", "") for m in o.json().get("models", [])]
            ok = any(MODEL == n or MODEL in n for n in names)
            print("Model:", MODEL, "FOUND" if ok else f"NOT FOUND (available: {names})")
            return ok
    except Exception as exc:
        print("Preflight failed:", type(exc).__name__, exc)
        return False

def request(client: httpx.Client, case: dict) -> dict:
    payload = {"message": case["message"], "user_id": USER}
    if case.get("document_reference"):
        payload["document_reference"] = str(CORPUS / case["document_reference"])
        payload["auto_index"] = True
    started = time.perf_counter()
    try:
        response = client.post(f"{API}/api/v1/chat", json=payload, timeout=300)
        elapsed = round(time.perf_counter() - started, 3)
        if response.status_code != 200:
            return {"passed": False, "status_code": response.status_code, "latency_seconds": elapsed, "error": response.text}
        return {"passed": None, "status_code": 200, "latency_seconds": elapsed, "body": response.json()}
    except Exception as exc:
        return {"passed": False, "latency_seconds": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"}

def evaluate(case: dict, raw: dict) -> dict:
    if raw.get("passed") is False:
        return {**raw, "id": case["id"], "reasons": [raw.get("error", "request failed")]}
    body = raw["body"]
    text = str(body.get("response", ""))
    low = text.lower()
    md = body.get("metadata") or {}
    reasons: list[str] = []
    passed = True

    if body.get("intent") != case["expected_intent"]:
        passed = False; reasons.append(f"intent={body.get('intent')!r}, expected={case['expected_intent']!r}")
    def _normalized_text(value: str) -> str:
        # Make evaluator matching robust to presentation formatting such as
        # currency symbols and thousands separators (e.g. $14,400 vs 14400).
        import re
        value = value.lower()
        value = re.sub(r"(?<=\d)[,\s](?=\d)", "", value)
        value = re.sub(r"[$€£₹]", "", value)
        return value

    normalized_low = _normalized_text(text)
    for term in case.get("expected_terms", []):
        term_low = term.lower()
        if term_low not in low and _normalized_text(term_low) not in normalized_low:
            passed = False; reasons.append(f"missing expected term: {term}")
    for term in case.get("forbidden_terms", []):
        if term.lower() in low:
            passed = False; reasons.append(f"forbidden claim: {term}")
    if case.get("require_evidence"):
        # For answerable cases, evidence is mandatory. For an explicitly
        # unanswerable case, zero evidence is valid; an unverified response
        # is also acceptable when it clearly refuses to invent the fact.
        if md.get("evidence_count", 0) <= 0 and not case.get("unanswerable"):
            passed = False; reasons.append("no evidence_count")
        allowed_statuses = {"ok", "indexed", "not_found", "no_context", "no_retrieval", "ambiguous", "discovered_not_indexed"}
        if case.get("unanswerable"):
            allowed_statuses.add("unverified")
        if md.get("status") not in allowed_statuses:
            passed = False; reasons.append(f"unexpected RAG status: {md.get('status')!r}")
    if case.get("unanswerable"):
        refusal_markers = ("not found", "not available", "cannot", "can't", "do not have", "no evidence", "unable")
        if not any(x in low for x in refusal_markers):
            passed = False; reasons.append("unanswerable request did not clearly refuse/qualify the missing fact")
    if case.get("require_plan"):
        if not body.get("plan"): passed = False; reasons.append("missing plan")
        if not isinstance(md.get("step_count"), int) or md.get("step_count", 0) < 1:
            passed = False; reasons.append("missing/invalid step_count")
    if case.get("require_application_execution"):
        action = md.get("application_action") or {}
        execution = md.get("execution") or {}
        if action.get("operation") != "inspect":
            passed = False; reasons.append(f"expected inspect operation, got {action.get('operation')!r}")
        if execution.get("status") != "executed":
            passed = False; reasons.append(f"application execution status={execution.get('status')!r}")
    if case.get("safety") == "fail_closed":
        if any(x in low for x in ("closed successfully", "file closed", "application closed")):
            passed = False; reasons.append("destructive/unsupported close appears executed")
        action = md.get("execution") or {}
        if action.get("status") == "executed":
            passed = False; reasons.append("application executor reported executed for close")

    return {
        "id": case["id"], "passed": passed, "status_code": 200,
        "latency_seconds": raw["latency_seconds"], "intent": body.get("intent"),
        "selected_agent": body.get("selected_agent"), "response": body.get("response", ""),
        "metadata": md, "reasons": reasons,
    }

def memory_cases() -> list[dict]:
    return [
        {"id": "MEM-STORE", "message": "Remember that my primary Phoenix AI vector database is Qdrant.", "expected_intent": "memory", "expected_terms": []},
        {"id": "MEM-RECALL", "message": "What vector database did I say is my primary choice for Phoenix AI?", "expected_intent": "memory", "expected_terms": ["qdrant"]},
    ]

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    if not preflight(): return 2
    if args.preflight: return 0

    cases = json.loads(DATA.read_text(encoding="utf-8"))
    results = []
    with httpx.Client() as client:
        for case in cases + memory_cases():
            print(f"\n[{case['id']}] {case['message']}")
            result = evaluate(case, request(client, case)); results.append(result)
            print("PASS" if result["passed"] else "FAIL", result.get("reasons", []), f"({result.get('latency_seconds')}s)")

    passed = sum(bool(x.get("passed")) for x in results)
    total = len(results)
    out = {
        "suite": "Phoenix AI Real-World E2E Evaluation v3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": passed, "total": total,
        "pass_rate": round(passed / total, 4) if total else 0,
        "results": results,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Phoenix AI Real-World E2E Evaluation v3", "", f"**Result:** {passed}/{total} passed ({out['pass_rate']:.1%})", ""]
    for x in results:
        md += [f"## {x['id']}: {'PASS' if x.get('passed') else 'FAIL'}", f"- Latency: {x.get('latency_seconds')}s", f"- Intent: {x.get('intent')}", f"- Agent: {x.get('selected_agent')}"]
        if x.get("reasons"): md.append("- Reasons: " + "; ".join(x["reasons"]))
    (RESULTS / "latest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nFINAL: {passed}/{total} passed")
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
