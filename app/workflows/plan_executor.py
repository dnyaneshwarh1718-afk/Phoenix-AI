from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.agents.base_agent import AgentContext, AgentResult
from app.planning.models import Plan, PlanStep


@dataclass(frozen=True)
class StepExecution:
    step_id: str
    agent: str
    status: str
    success: bool
    response: str
    data: dict[str, Any]
    error: str | None = None
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent": self.agent,
            "status": self.status,
            "success": self.success,
            "response": self.response,
            "data": self.data,
            "error": self.error,
            "latency_seconds": round(self.latency_seconds, 6),
        }


class PlanExecutor:
    """Execute a validated Phoenix plan with dependency and safety boundaries.

    Planning remains separate from execution. The executor accepts only an
    already validated Plan and dispatches each step to the corresponding
    specialized agent. Application actions remain protected by the existing
    approval manager because the executor never bypasses an agent's safety
    boundary.
    """

    EXECUTABLE_AGENTS = {"rag", "research", "memory", "application", "vision", "general"}

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    async def execute(self, plan: Plan | dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        plan_obj = self._coerce_plan(plan)
        valid, errors = plan_obj.validate()
        if not valid:
            return {
                "status": "invalid_plan",
                "success": False,
                "steps": [],
                "errors": errors,
            }

        completed: dict[str, StepExecution] = {}
        pending = list(plan_obj.steps)
        step_timings: dict[str, float] = {}

        while pending:
            ready = [
                step for step in pending
                if all(dep in completed and completed[dep].success for dep in step.depends_on)
            ]

            if not ready:
                blocked = [
                    step.id for step in pending
                    if any(dep in completed and not completed[dep].success for dep in step.depends_on)
                ]
                return {
                    "status": "blocked",
                    "success": False,
                    "steps": [item.to_dict() for item in completed.values()],
                    "blocked_steps": blocked or [step.id for step in pending],
                    "errors": ["Plan dependencies could not be satisfied."],
                    "step_timings": step_timings,
                }

            # Preserve plan order for deterministic local execution.
            for step in ready:
                pending.remove(step)
                execution = await self._execute_step(step, state, completed)
                completed[step.id] = execution
                step_timings[step.id] = execution.latency_seconds

                # Fail closed: do not execute dependent steps after a failed,
                # unverified, or blocked step.
                if not execution.success:
                    return {
                        "status": execution.status,
                        "success": False,
                        "steps": [item.to_dict() for item in completed.values()],
                        "blocked_steps": [item.id for item in pending],
                        "errors": [execution.error] if execution.error else [],
                        "step_timings": step_timings,
                    }

        return {
            "status": "completed",
            "success": True,
            "steps": [item.to_dict() for item in completed.values()],
            "step_timings": step_timings,
        }

    async def _execute_step(
        self,
        step: PlanStep,
        state: dict[str, Any],
        completed: dict[str, StepExecution],
    ) -> StepExecution:
        started = perf_counter()

        if step.agent == "planning":
            return StepExecution(
                step.id, step.agent, "blocked", False, "", {},
                "Nested planning execution is not permitted inside a plan.",
                perf_counter() - started,
            )

        if step.agent not in self.EXECUTABLE_AGENTS:
            return StepExecution(
                step.id, step.agent, "blocked", False, "", {},
                f"Unsupported execution agent: {step.agent}",
                perf_counter() - started,
            )

        metadata = dict(state.get("metadata") or {})
        prior_results = [item.to_dict() for item in completed.values()]
        metadata.update({
            "plan_step_id": step.id,
            "plan_step_agent": step.agent,
            "plan_step_tools": list(step.tools),
            "plan_step_approval_required": step.approval_required,
            # Closed-loop context: downstream steps can consume verified
            # outputs from upstream steps without inventing prior results.
            "plan_prior_results": prior_results,
        })

        # Carry the user's explicit document reference into every plan step.
        # Application steps need the target path, while RAG/structured agents
        # use the same reference for document-scoped retrieval. This prevents
        # the planner from losing the target merely because the plan was
        # generated between routing and execution.
        document_reference = state.get("metadata", {}).get("document_reference")
        if document_reference:
            metadata.setdefault("document_reference", document_reference)
            metadata.setdefault("target", document_reference)
            metadata.setdefault("file_path", document_reference)

        context = AgentContext(
            request_id=state["request_id"],
            user_id=state.get("user_id", "default"),
            project_id=state.get("project_id"),
            metadata=metadata,
            memory_context=list(state.get("memory_context") or []),
        )

        try:
            if step.agent == "general":
                temp_state = dict(state)
                temp_state["metadata"] = metadata
                temp_state["user_message"] = step.description
                result = await self.orchestrator.execute_general(temp_state)
                agent_result = AgentResult(
                    success=bool(result.get("response")),
                    content=result.get("response", ""),
                    data=dict(result.get("metadata") or {}),
                )
            else:
                agent_result = await self.orchestrator.agents[step.agent].run(
                    step.description,
                    context,
                )
        except Exception as exc:
            return StepExecution(
                step.id, step.agent, "failed", False, "", {},
                f"{type(exc).__name__}: {exc}", perf_counter() - started,
            )

        execution_data = agent_result.data.get("execution") if isinstance(agent_result.data, dict) else None
        status = str(
            agent_result.data.get("status")
            or (execution_data or {}).get("status")
            or "ok"
        )
        success = bool(agent_result.success)

        # Explicit safety blocks are successful from the safety-contract
        # perspective only when the agent says nothing was executed.
        if status == "blocked" and agent_result.data.get("execution", {}).get("executed") is False:
            success = True

        if status == "unverified":
            success = False

        return StepExecution(
            step.id,
            step.agent,
            status,
            success,
            agent_result.content,
            dict(agent_result.data or {}),
            agent_result.error,
            perf_counter() - started,
        )

    @staticmethod
    def _coerce_plan(value: Plan | dict[str, Any]) -> Plan:
        if isinstance(value, Plan):
            return value
        if not isinstance(value, dict):
            raise TypeError("Plan must be a Plan object or dictionary.")

        from app.planning.models import PlanStatus

        steps = []
        for raw in value.get("steps", []):
            steps.append(
                PlanStep(
                    id=str(raw.get("id", "")).strip(),
                    title=str(raw.get("title", "")).strip(),
                    description=str(raw.get("description", "")).strip(),
                    agent=str(raw.get("agent", "general")).strip().lower(),
                    depends_on=tuple(str(v) for v in raw.get("depends_on", []) or []),
                    tools=tuple(str(v) for v in raw.get("tools", []) or []),
                    approval_required=bool(raw.get("approval_required", False)),
                )
            )
        raw_status = str(value.get("status", "ready"))
        try:
            status = PlanStatus(raw_status)
        except ValueError:
            status = PlanStatus.READY
        return Plan(
            goal=str(value.get("goal", "")),
            steps=tuple(steps),
            status=status,
            blockers=tuple(str(v) for v in value.get("blockers", []) or []),
            assumptions=tuple(str(v) for v in value.get("assumptions", []) or []),
            notes=tuple(str(v) for v in value.get("notes", []) or []),
            metadata=dict(value.get("metadata", {}) or {}),
        )
