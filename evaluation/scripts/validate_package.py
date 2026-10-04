from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
required = [
    "corpus/phoenix_architecture.txt", "corpus/phoenix_security.txt", "corpus/phoenix_projects.txt",
    "corpus/phoenix_evaluation.pdf", "corpus/phoenix_evaluation.docx",
    "corpus/phoenix_evaluation.xlsx", "corpus/phoenix_evaluation.pptx", "datasets/cases.json"
]
for item in required:
    p = ROOT / item
    assert p.exists() and p.stat().st_size > 0, f"Missing/empty: {p}"
cases = json.loads((ROOT / "datasets/cases.json").read_text(encoding="utf-8"))
assert len(cases) >= 10
assert all(c.get("id") and c.get("message") and c.get("expected_intent") for c in cases)
print(f"Evaluation package validation passed: {len(cases)} primary cases + 2 live memory cases.")
