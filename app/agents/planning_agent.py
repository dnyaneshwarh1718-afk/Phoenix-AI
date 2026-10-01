from __future__ import annotations

import json
import re
from typing import Any

from app.agents.base_agent import AgentContext, AgentResult, BaseAgent
from app.llm.gateway import LLMGateway
from app.llm.models import ModelRequest
from app.planning.models import Plan, PlanStatus, PlanStep


class PlanningAgent(BaseAgent):
    """Create validated, executable task plans without executing tools."""

    name = "planning"
    SUPPORTED_AGENTS = {"general", "rag", "research", "planning", "memory", "application", "vision"}

    def __init__(self, llm: LLMGateway, *, max_steps: int = 12):
        self.llm = llm
        self.max_steps = max(1, min(30, max_steps))

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        goal = (message or "").strip()
        if not goal:
            return AgentResult(False, error="Planning goal cannot be empty.", data={"status": "invalid"})

        prompt = self._build_prompt(goal, context)
        try:
            response = await self.llm.chat(
                ModelRequest(
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    task_type="reasoning",
                    temperature=0.1,
                )
            )
        except Exception as exc:
            # A planning failure must never be silently presented as a valid plan.
            return AgentResult(
                False,
                content="I could not create a reliable execution plan because the planning model was unavailable.",
                data={"status": "llm_failed"},
                error=f"Planning model failed: {exc}",
            )

        try:
            plan = self._parse_plan(response.content, goal)
        except ValueError as exc:
            return AgentResult(
                False,
                content="The planning model returned an invalid execution plan.",
                data={"status": "invalid_plan", "llm_provider": response.provider, "llm_model": response.model},
                error=str(exc),
            )

        valid, errors = plan.validate()
        if not valid:
            return AgentResult(
                False,
                content="The generated execution plan failed validation and was not approved for execution.",
                data={"status": "invalid_plan", "plan": plan.to_dict(), "validation_errors": errors},
                error="; ".join(errors),
            )

        rendered = self._render(plan)
        return AgentResult(
            True,
            content=rendered,
            data={
                "status": plan.status.value,
                "plan": plan.to_dict(),
                "step_count": len(plan.steps),
                "llm_provider": response.provider,
                "llm_model": response.model,
            },
        )

    @staticmethod
    def _system_prompt() -> str:
        return """You are Phoenix AI's Planning Agent.
Create a safe, concrete execution plan for the user's goal. Do not execute tools.
Return ONLY valid JSON, with this exact top-level shape:
{
  "goal": "...",
  "steps": [
    {"id":"step_1","title":"...","description":"...","agent":"general|rag|research|planning|memory|application|vision","depends_on":[],"tools":[],"approval_required":false}
  ],
  "blockers": [],
  "assumptions": [],
  "notes": []
}
Rules:
- Use 1-12 steps.
- IDs must be step_1, step_2, ... and dependencies must reference existing IDs.
- Keep dependencies minimal and acyclic.
- Choose the specialized agent only when clearly appropriate.
- Mark destructive/external side effects as approval_required=true.
- If essential information is missing, put it in blockers instead of inventing it.
- Do not invent tool capabilities or claim an action was executed."""

    def _build_prompt(self, goal: str, context: AgentContext) -> str:
        metadata = context.metadata or {}
        available = metadata.get("available_tools", [])
        return (
            f"User goal:\n{goal}\n\n"
            f"Available tools/context:\n{json.dumps(available, ensure_ascii=False, default=str)}\n\n"
            "Produce the JSON execution plan now."
        )

    def _parse_plan(self, content: str, fallback_goal: str) -> Plan:
        raw = self._extract_json(content)
        if not isinstance(raw, dict):
            raise ValueError("Planner output must be a JSON object.")

        goal = str(raw.get("goal") or fallback_goal).strip()
        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError("Planner output must contain a steps list.")
        if not raw_steps:
            raise ValueError("Planner returned no steps.")
        if len(raw_steps) > self.max_steps:
            raise ValueError(f"Planner returned {len(raw_steps)} steps; maximum is {self.max_steps}.")

        steps: list[PlanStep] = []
        for index, item in enumerate(raw_steps, 1):
            if not isinstance(item, dict):
                raise ValueError(f"Step {index} is not an object.")
            sid = str(item.get("id") or f"step_{index}").strip()
            title = str(item.get("title") or "").strip()
            description = str(item.get("description") or "").strip()
            agent = str(item.get("agent") or "general").strip().lower()
            if agent not in self.SUPPORTED_AGENTS:
                raise ValueError(f"Step {sid} references unsupported agent: {agent}.")
            deps = self._string_tuple(item.get("depends_on", []))
            tools = self._string_tuple(item.get("tools", []))
            raw_approval = item.get("approval_required", False)
            if isinstance(raw_approval, bool):
                approval = raw_approval
            elif isinstance(raw_approval, str) and raw_approval.strip().lower() in {"true", "false"}:
                approval = raw_approval.strip().lower() == "true"
            else:
                raise ValueError(f"Step {sid} has invalid approval_required value.")
            steps.append(PlanStep(sid, title, description, agent, deps, tools, approval))

        blockers = self._string_tuple(raw.get("blockers", []))
        assumptions = self._string_tuple(raw.get("assumptions", []))
        notes = self._string_tuple(raw.get("notes", []))
        status = PlanStatus.BLOCKED if blockers else PlanStatus.READY
        return Plan(goal, tuple(steps), status, blockers, assumptions, notes)

    @staticmethod
    def _extract_json(content: str) -> Any:
        text = (content or "").strip()
        # Handle markdown fences and models that prepend a short explanation.
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        candidate = fenced.group(1).strip() if fenced else text
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Planner JSON could not be parsed: {exc}") from exc
            raise ValueError("Planner did not return JSON.")

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value.strip(),) if value.strip() else ()
        if not isinstance(value, list):
            return ()
        return tuple(str(x).strip() for x in value if str(x).strip())

    @staticmethod
    def _render(plan: Plan) -> str:
        lines = [f"Goal: {plan.goal}", f"Status: {plan.status.value}", "", "Execution plan:"]
        for step in plan.steps:
            deps = f" | depends on: {', '.join(step.depends_on)}" if step.depends_on else ""
            approval = " | approval required" if step.approval_required else ""
            lines.append(f"{step.id}. {step.title} [{step.agent}]\n   {step.description}{deps}{approval}")
        if plan.blockers:
            lines.extend(["", "Blockers:", *[f"- {b}" for b in plan.blockers]])
        if plan.assumptions:
            lines.extend(["", "Assumptions:", *[f"- {a}" for a in plan.assumptions]])
        return "\n".join(lines)
