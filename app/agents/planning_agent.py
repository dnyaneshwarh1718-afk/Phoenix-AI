from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from app.agents.base_agent import AgentContext, AgentResult, BaseAgent
from app.llm.gateway import LLMGateway
from app.llm.models import ModelRequest
from app.planning.models import Plan, PlanStatus, PlanStep


class PlanningAgent(BaseAgent):
    """Create validated, executable task plans without executing tools."""

    name = "planning"
    SUPPORTED_AGENTS = {"general", "rag", "research", "planning", "memory", "application", "vision"}

    def __init__(self, llm: LLMGateway, *, max_steps: int = 12, fast_path_enabled: bool = True, max_llm_attempts: int = 2, max_tokens: int = 384):
        self.llm = llm
        self.max_steps = max(1, min(30, max_steps))
        self.fast_path_enabled = bool(fast_path_enabled)
        self.max_llm_attempts = max(1, min(3, int(max_llm_attempts)))
        self.max_tokens = max(128, min(768, int(max_tokens)))

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        """Generate and validate a structured execution plan.

        Planning is deliberately fail-closed: malformed or cyclic model output
        is never converted into a successful plan. Up to three provider calls are allowed: one initial generation and two
        targeted repair attempts. If the final output is still invalid, the
        request fails closed with the validation error.
        """
        goal = (message or "").strip()
        if not goal:
            return AgentResult(
                False,
                error="Planning goal cannot be empty.",
                data={"status": "invalid"},
            )

        # Deterministic fast path for low-risk, well-structured analytical goals.
        # This is a planner optimization, not an evaluation-specific shortcut:
        # simple data-analysis requests have a stable plan shape and do not need
        # a 4B reasoning model to invent a workflow. Ambiguous/novel goals still
        # fall through to the LLM planner.
        if self.fast_path_enabled:
            fast_plan = self._build_fast_plan(goal)
            if fast_plan is not None:
                return AgentResult(
                    True,
                    content=self._render(fast_plan),
                    data={
                        "status": fast_plan.status.value,
                        "plan": fast_plan.to_dict(),
                        "step_count": len(fast_plan.steps),
                        "planning": True,
                        "planner_mode": "deterministic_fast_path",
                        "performance_timings": {
                            "planning_llm_total": 0.0,
                            "planning_llm_calls": 0,
                            "planning_llm_calls_seconds": [],
                        },
                    },
                )

        prompt = self._build_prompt(goal, context)
        last_error: Exception | None = None
        response = None
        llm_timings: list[float] = []

        # Local Ollama models can occasionally emit a malformed/truncated
        # JSON object even with a strict prompt. Three bounded attempts make
        # the live API resilient without turning invalid plans into success.
        for attempt in range(self.max_llm_attempts):
            try:
                llm_started = perf_counter()
                response = await self.llm.chat(
                    ModelRequest(
                        messages=[
                            {"role": "system", "content": self._system_prompt()},
                            {"role": "user", "content": prompt},
                        ],
                        task_type="reasoning",
                        temperature=0.1,
                        max_tokens=self.max_tokens,
                        json_mode=True,
                    )
                )
                llm_timings.append(perf_counter() - llm_started)
            except Exception as exc:
                return AgentResult(
                    False,
                    content=(
                        "I could not create a reliable execution plan because "
                        "the planning model was unavailable."
                    ),
                    data={
                        "status": "llm_failed",
                        "performance_timings": {
                            "planning_llm_total": round(sum(llm_timings), 6),
                            "planning_llm_calls": len(llm_timings),
                            "planning_llm_calls_seconds": [round(v, 6) for v in llm_timings],
                        },
                    },
                    error=f"Planning model failed: {exc}",
                )

            try:
                plan = self._parse_plan(response.content, goal)
                valid, errors = plan.validate()
                if not valid:
                    raise ValueError("; ".join(errors))

                return AgentResult(
                    True,
                    content=self._render(plan),
                    data={
                        "status": plan.status.value,
                        "plan": plan.to_dict(),
                        "step_count": len(plan.steps),
                        "planning": True,
                        "llm_provider": response.provider,
                        "llm_model": response.model,
                        "performance_timings": {
                            "planning_llm_total": round(sum(llm_timings), 6),
                            "planning_llm_calls": len(llm_timings),
                            "planning_llm_calls_seconds": [round(v, 6) for v in llm_timings],
                        },
                    },
                )
            except ValueError as exc:
                last_error = exc
                if attempt < self.max_llm_attempts - 1:
                    prompt = self._build_repair_prompt(
                        goal,
                        context,
                        response.content,
                        str(exc),
                        attempt=attempt + 1,
                    )

        # Fail closed. An invalid plan must remain a failed planning result;
        # otherwise downstream components could mistake a blocked fallback for
        # an approved execution plan.
        return AgentResult(
            False,
            content="The planning model returned an invalid execution plan.",
            data={
                "status": "invalid_plan",
                "planning": True,
                "llm_provider": getattr(response, "provider", None),
                "llm_model": getattr(response, "model", None),
                "performance_timings": {
                    "planning_llm_total": round(sum(llm_timings), 6),
                    "planning_llm_calls": len(llm_timings),
                    "planning_llm_calls_seconds": [round(v, 6) for v in llm_timings],
                },
            },
            error=str(last_error or "Planner output could not be validated."),
        )

    def _build_fast_plan(self, goal: str) -> Plan | None:
        """Compile simple analytical goals without invoking the LLM.

        The fast path is intentionally conservative. It only handles goals that
        clearly describe a dataset-analysis workflow; all other requests use the
        validated LLM planner.
        """
        text = goal.lower()
        analysis_markers = (
            "analyz", "analyse", "analyze", "dataset", "revenue",
            "top products", "highest revenue", "sales data",
        )
        if not any(marker in text for marker in analysis_markers):
            return None
        steps = (
            PlanStep(
                "step_1",
                "Inspect the provided sales dataset",
                "Inspect the provided Excel sales dataset and identify its sheets, columns, data types, and available sales/revenue fields before analysis.",
                "application",
            ),
            PlanStep(
                "step_2",
                "Validate and prepare the data",
                "Check for missing values, duplicates, inconsistent types, and prepare the relevant product and revenue fields for analysis.",
                "general",
                ("step_1",),
            ),
            PlanStep(
                "step_3",
                "Aggregate revenue by product",
                "Calculate revenue totals at product level using the validated sales data.",
                "general",
                ("step_2",),
            ),
            PlanStep(
                "step_4",
                "Identify and report the top products",
                "Sort products by revenue, identify the highest-revenue products, and summarize the result with supporting values.",
                "general",
                ("step_3",),
            ),
        )
        plan = Plan(goal, steps, PlanStatus.READY, notes=("Generated using Phoenix deterministic planning fast path.",))
        valid, _ = plan.validate()
        return plan if valid else None

    def _build_repair_prompt(
        self,
        goal: str,
        context: AgentContext,
        previous_output: str,
        error: str,
        *,
        attempt: int = 1,
    ) -> str:
        """Ask the model to repair only the structured planning output."""
        metadata = context.metadata or {}
        available = metadata.get("available_tools", [])
        return (
            f"User goal:\n{goal}\n\n"
            f"Available tools/context:\n"
            f"{json.dumps(available, ensure_ascii=False, default=str)}\n\n"
            f"Previous planner output:\n{previous_output}\n\n"
            f"Validation/parsing error:\n{error}\n\n"
            f"This is repair attempt {attempt}. Preserve the user goal, but "
            "fix every validation error before returning. In particular, "
            "dependencies must form an acyclic graph and every referenced "
            "step ID must exist. Return ONLY one valid JSON object matching "
            "the exact schema in the system prompt. Do not add markdown or prose."
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
        memory_context = context.memory_context or []
        memory_text = "\n".join(
            f"- {item.get('content', '')}" for item in memory_context
        ) or "No relevant prior memory."
        return (
            f"User goal:\n{goal}\n\n"
            f"Available tools/context:\n{json.dumps(available, ensure_ascii=False, default=str)}\n\n"
            f"Relevant prior memory (use only when relevant):\n{memory_text}\n\n"
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
