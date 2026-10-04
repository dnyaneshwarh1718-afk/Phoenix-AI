"""Dependency-free architectural guard for the application-control boundary."""
from pathlib import Path
import ast

source = Path("app/orchestrator/orchestrator.py").read_text(encoding="utf-8")
tree = ast.parse(source)
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        imports.append((node.module, tuple(alias.name for alias in node.names)))

assert ("app.execution.approval", ("ApprovalManager",)) in imports, (
    "PhoenixOrchestrator must import ApprovalManager from app.execution.approval"
)
assert not any(
    module == "app.core.security" and "ApprovalManager" in names
    for module, names in imports
), "ApprovalManager must not be imported from app.core.security"

print("ARCHITECTURE_CHECK=PASS")
