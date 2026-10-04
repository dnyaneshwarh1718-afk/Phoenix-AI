from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.agents.base_agent import AgentContext, AgentResult, BaseAgent
from app.core.security import SecurityPolicy
from app.execution.approval import ApprovalManager


class ApplicationExecutor(Protocol):
    """Side-effect boundary for desktop/application operations."""

    def execute(self, action: "ApplicationAction") -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ApplicationAction:
    operation: str
    application: str
    target: str | None = None
    arguments: tuple[str, ...] = ()
    approved: bool = False


APP_ALIASES = {
    "excel": "excel",
    "microsoft excel": "excel",
    "word": "word",
    "microsoft word": "word",
    "powerpoint": "powerpoint",
    "power point": "powerpoint",
    "ppt": "powerpoint",
    "power bi": "power_bi",
    "powerbi": "power_bi",
    "jupyter": "jupyter",
    "jupyter notebook": "jupyter",
}

EXTENSIONS = {
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".csv": "excel",
    ".docx": "word",
    ".doc": "word",
    ".pptx": "powerpoint",
    ".ppt": "powerpoint",
    ".ipynb": "jupyter",
}


class WindowsApplicationExecutor:
    """
    Windows desktop executor.

    Safe actions such as opening an existing file are allowed.
    High-risk actions are gated by ApprovalManager before this layer is reached.
    """

    def __init__(self, approval: ApprovalManager | None = None):
        self.approval = approval or ApprovalManager(SecurityPolicy())

    def execute(self, action: ApplicationAction) -> dict[str, Any]:
        if action.operation in {"delete", "send", "upload", "execute_shell", "modify"}:
            self.approval.check(action.operation, approved=action.approved)

        if action.operation in {"open", "launch"}:
            return self._open(action)

        if action.operation == "inspect":
            return self._inspect(action)

        if action.operation == "close":
            return {
                "status": "unsupported",
                "message": "Closing desktop applications is not enabled yet.",
            }

        return {
            "status": "unsupported",
            "message": f"Unsupported application operation: {action.operation}",
        }

    def _open(self, action: ApplicationAction) -> dict[str, Any]:
        target = action.target

        if target:
            path = Path(target).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Target does not exist: {path}")

            if platform.system() == "Windows":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])

            return {
                "status": "executed",
                "operation": "open",
                "application": action.application,
                "target": str(path.resolve()),
            }

        # Launch the application itself when no target was supplied.
        commands = {
            "excel": ["excel"],
            "word": ["winword"],
            "powerpoint": ["powerpnt"],
            "power_bi": ["pbidesktop"],
            "jupyter": ["jupyter", "notebook"],
        }
        command = commands.get(action.application)
        if not command:
            raise ValueError(f"No launch command registered for {action.application}")

        subprocess.Popen(command)
        return {
            "status": "executed",
            "operation": "launch",
            "application": action.application,
        }

    def _inspect(self, action: ApplicationAction) -> dict[str, Any]:
        target = action.target
        if not target:
            raise ValueError("Inspect requires a target file.")

        path = Path(target).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Target does not exist: {path}")

        if path.suffix.lower() in {".xlsx", ".xlsm"}:
            try:
                from openpyxl import load_workbook
            except ImportError as exc:
                raise RuntimeError(
                    "openpyxl is required for Excel inspection. Install it with: pip install openpyxl"
                ) from exc

            workbook = load_workbook(path, read_only=True, data_only=True)
            sheets = {}
            for ws in workbook.worksheets:
                sheets[ws.title] = {
                    "rows": ws.max_row,
                    "columns": ws.max_column,
                }
            workbook.close()
            return {
                "status": "executed",
                "operation": "inspect",
                "application": "excel",
                "target": str(path.resolve()),
                "sheets": sheets,
            }

        return {
            "status": "executed",
            "operation": "inspect",
            "application": action.application,
            "target": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }


class ApplicationActionParser:
    """Deterministic parser; no LLM is required to decide whether a desktop action is requested."""

    _open = re.compile(r"\b(open|launch|start|run)\b", re.I)
    _inspect = re.compile(r"\b(analy[sz]e|inspect|read|summari[sz]e|profile)\b", re.I)
    _close = re.compile(r"\b(close|quit|exit)\b", re.I)

    def parse(self, message: str, metadata: dict[str, Any] | None = None) -> ApplicationAction:
        text = (message or "").strip()
        lower = text.lower()
        metadata = metadata or {}

        app = self._detect_application(lower, metadata)
        if not app:
            raise ValueError(
                "No supported desktop application detected. "
                "Supported applications: Excel, Word, PowerPoint, Power BI, Jupyter."
            )

        target = self._detect_target(text, metadata)

        if self._inspect.search(text):
            operation = "inspect"
        elif self._close.search(text):
            operation = "close"
        elif self._open.search(text):
            operation = "open"
        else:
            operation = "launch"

        approved = bool(metadata.get("application_approved", False))
        return ApplicationAction(
            operation=operation,
            application=app,
            target=target,
            approved=approved,
        )

    def _detect_application(self, text: str, metadata: dict[str, Any]) -> str | None:
        explicit = metadata.get("application")
        if isinstance(explicit, str):
            key = explicit.strip().lower()
            if key in APP_ALIASES:
                return APP_ALIASES[key]

        for alias, canonical in sorted(APP_ALIASES.items(), key=lambda x: -len(x[0])):
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
                return canonical

        target = metadata.get("target")
        if isinstance(target, str):
            detected = EXTENSIONS.get(Path(target).suffix.lower())
            if detected:
                return detected

        # Infer the application from a filename mentioned in the request.
        for ext, canonical in EXTENSIONS.items():
            if re.search(rf"(?<![A-Za-z0-9_])[^\s,;]+{re.escape(ext)}(?![A-Za-z0-9_])", text):
                return canonical

        return None

    def _detect_target(self, message: str, metadata: dict[str, Any]) -> str | None:
        explicit = metadata.get("target") or metadata.get("file_path")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()

        # Prefer an explicit quoted path.
        quoted = re.findall(r'["\']([^"\']+\.[A-Za-z0-9]{1,8})["\']', message)
        if quoted:
            return quoted[0]

        # Conservative path detection; avoids treating ordinary words as paths.
        candidates = re.findall(
            r"(?:(?:[A-Za-z]:[\\/])|(?:\./|\.{2}/)|(?:[\\/]))[^\s,;]+",
            message,
        )
        for candidate in candidates:
            clean = candidate.rstrip(".,)")
            if Path(clean).suffix.lower() in EXTENSIONS:
                return clean

        # Also accept a bare filename such as ``sales.xlsx``.
        filenames = re.findall(
            r"(?<![A-Za-z0-9_./\\-])([A-Za-z0-9_.-]+\.(?:xlsx|xlsm|xls|csv|docx|doc|pptx|ppt|ipynb))(?![A-Za-z0-9_])",
            message,
            flags=re.I,
        )
        if filenames:
            return filenames[0].strip()

        return None


class ApplicationControlAgent(BaseAgent):
    """
    Phoenix desktop/application control agent.

    Architecture:
        user request -> deterministic action parser -> approval gate -> executor

    The executor is injectable, making the agent fully testable without launching
    real applications during unit tests.
    """

    name = "application"

    def __init__(
        self,
        executor: ApplicationExecutor | None = None,
        approval: ApprovalManager | None = None,
    ):
        policy = SecurityPolicy()
        self.approval = approval or ApprovalManager(policy)
        self.executor = executor or WindowsApplicationExecutor(self.approval)
        self.parser = ApplicationActionParser()

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        try:
            if context.metadata.get("application_enabled") is False:
                return AgentResult(
                    success=False,
                    content="Application Control is disabled by Phoenix AI configuration.",
                    data={"application_action": {"status": "disabled"}},
                    error="Application control is disabled.",
                )
            action = self.parser.parse(message, context.metadata)
            data = {
                "application_action": {
                    "operation": action.operation,
                    "application": action.application,
                    "target": action.target,
                }
            }

            result = self.executor.execute(action)
            data["execution"] = result

            status = result.get("status")
            if status == "executed":
                target = result.get("target")
                suffix = f" Target: {target}" if target else ""
                return AgentResult(
                    success=True,
                    content=(
                        f"Application action completed: {action.operation} "
                        f"{action.application}.{suffix}"
                    ),
                    data=data,
                )

            return AgentResult(
                success=True,
                content=result.get("message", "Application action was not executed."),
                data=data,
            )

        except Exception as exc:
            return AgentResult(
                success=False,
                content="The requested application action could not be completed.",
                data={
                    "application_action": {
                        "status": "failed",
                    }
                },
                error=f"{type(exc).__name__}: {exc}",
            )
