from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PlanStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    INVALID = "invalid"


@dataclass(frozen=True)
class PlanStep:
    id: str
    title: str
    description: str
    agent: str = "general"
    depends_on: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    approval_required: bool = False


@dataclass(frozen=True)
class Plan:
    goal: str
    steps: tuple[PlanStep, ...]
    status: PlanStatus = PlanStatus.READY
    blockers: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        ids = [step.id for step in self.steps]
        if not self.goal.strip():
            errors.append("Plan goal cannot be empty.")
        if not self.steps:
            errors.append("Plan must contain at least one step.")
        if len(ids) != len(set(ids)):
            errors.append("Plan step IDs must be unique.")
        known = set(ids)
        for step in self.steps:
            if not step.title.strip() or not step.description.strip():
                errors.append(f"Step {step.id} must have a title and description.")
            missing = [dep for dep in step.depends_on if dep not in known]
            if missing:
                errors.append(f"Step {step.id} has unknown dependencies: {missing}.")
            if step.id in step.depends_on:
                errors.append(f"Step {step.id} cannot depend on itself.")
        if not errors:
            # Kahn-style cycle detection.
            remaining = {step.id: set(step.depends_on) for step in self.steps}
            resolved: set[str] = set()
            while remaining:
                ready = [sid for sid, deps in remaining.items() if not (deps - resolved)]
                if not ready:
                    errors.append("Plan contains a dependency cycle.")
                    break
                resolved.update(ready)
                for sid in ready:
                    remaining.pop(sid)
        return not errors, errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status.value,
            "blockers": list(self.blockers),
            "assumptions": list(self.assumptions),
            "notes": list(self.notes),
            "steps": [
                {
                    "id": s.id,
                    "title": s.title,
                    "description": s.description,
                    "agent": s.agent,
                    "depends_on": list(s.depends_on),
                    "tools": list(s.tools),
                    "approval_required": s.approval_required,
                }
                for s in self.steps
            ],
            "metadata": dict(self.metadata),
        }
