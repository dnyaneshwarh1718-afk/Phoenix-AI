from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.base_agent import AgentContext
from app.orchestrator.state import PhoenixState


# ======================================================================
# GRAPH BUILDER
# ======================================================================


def build_phoenix_graph(orchestrator):
    """
    Build the Phoenix AI orchestration graph.

    Flow:

        START
          ↓
        classify
          ↓
        route
          ↓
        plan
          ↓
        execute
          ↓
        END

    The graph is responsible for orchestration only.

    Agents are responsible for their domain-specific work.
    """

    graph = StateGraph(PhoenixState)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    graph.add_node(
        "classify",
        _build_classify_node(orchestrator),
    )

    graph.add_node(
        "load_memory",
        _build_memory_node(orchestrator),
    )

    graph.add_node(
        "route",
        _route,
    )

    graph.add_node(
        "plan",
        _build_plan_node(orchestrator),
    )

    graph.add_node(
        "execute",
        _build_execute_node(orchestrator),
    )

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    graph.add_edge(
        START,
        "classify",
    )

    graph.add_edge(
        "classify",
        "load_memory",
    )

    graph.add_edge(
        "load_memory",
        "route",
    )

    graph.add_edge(
        "route",
        "plan",
    )

    graph.add_edge(
        "plan",
        "execute",
    )

    graph.add_edge(
        "execute",
        END,
    )

    return graph.compile()


# ======================================================================
# CLASSIFICATION
# ======================================================================


def _build_classify_node(orchestrator):
    """
    Create the intent-classification node.

    Classification is synchronous because IntentRouter.classify()
    is synchronous.
    """

    def classify(state: PhoenixState) -> PhoenixState:
        metadata = dict(
            state.get("metadata") or {}
        )

        intent = orchestrator.router.classify(
            state["user_message"],
            metadata,
        )

        return {
            **state,
            "intent": intent,
            "metadata": metadata,
        }

    return classify


# ======================================================================
# MEMORY
# ======================================================================


def _build_memory_node(orchestrator):
    """Retrieve relevant prior memory before planning/agent execution."""

    def load_memory(state: PhoenixState) -> PhoenixState:
        if not orchestrator.settings.memory_auto_capture:
            return {**state, "memory_context": []}

        try:
            records = orchestrator.memory_store.search(
                user_id=state.get("user_id", "default"),
                project_id=state.get("project_id"),
                query=state["user_message"],
                limit=orchestrator.settings.memory_max_results,
            )
            context = [
                {
                    "memory_id": record.memory_id,
                    "kind": record.kind,
                    "content": record.content,
                    "importance": record.importance,
                    "created_at": record.created_at,
                }
                for record in records
            ]
            metadata = dict(state.get("metadata") or {})
            metadata["memory_retrieved"] = len(context)
            return {**state, "memory_context": context, "metadata": metadata}
        except Exception as exc:
            metadata = dict(state.get("metadata") or {})
            metadata["memory_retrieved"] = 0
            metadata["memory_status"] = "unavailable"
            metadata["memory_error"] = f"{type(exc).__name__}: {exc}"
            return {**state, "memory_context": [], "metadata": metadata}

    return load_memory


# ======================================================================
# ROUTING
# ======================================================================


def _route(state: PhoenixState) -> PhoenixState:
    """
    Map the classified intent to a Phoenix agent.
    """

    intent = state.get(
        "intent",
        "general",
    )

    return {
        **state,
        "selected_agent": intent,
    }


# ======================================================================
# PLANNING
# ======================================================================


def _build_plan_node(orchestrator):
    """
    Create the planning node.

    Planning occurs when:

    1. The router explicitly classifies the request as planning.
    2. A non-RAG request contains clear complexity markers.

    RAG requests intentionally bypass planning.
    """

    async def plan(state: PhoenixState) -> PhoenixState:

        intent = state.get(
            "intent",
            "general",
        )

        # Explicit planning request.
        if intent == "planning":
            return await _run_planning(
                orchestrator,
                state,
            )

        # RAG requests should go directly to retrieval.
        if intent == "rag":
            return state

        # Complex requests may benefit from planning.
        if _requires_planning(
            state["user_message"]
        ):
            return await _run_planning(
                orchestrator,
                state,
            )

        return state

    return plan


async def _run_planning(
    orchestrator,
    state: PhoenixState,
) -> PhoenixState:
    """
    Execute PlanningAgent and normalize its result.
    """

    result = await orchestrator.agents[
        "planning"
    ].run(
        state["user_message"],
        _agent_context(state),
    )

    return _normalize_planning_result(
        state,
        result,
    )


def _requires_planning(
    message: str,
) -> bool:
    """
    Determine whether a non-planning request is sufficiently complex
    to benefit from a planning pass.

    This deliberately remains conservative to avoid unnecessary LLM
    calls for simple requests.
    """

    message = message.lower()

    markers = (
        " and ",
        " then ",
        "after that",
        "create",
        "build",
        "analyze",
    )

    return any(
        marker in message
        for marker in markers
    )


# ======================================================================
# EXECUTION
# ======================================================================


def _build_execute_node(orchestrator):
    """
    Create the execution node.
    """

    async def execute(
        state: PhoenixState,
    ) -> PhoenixState:

        intent = state.get(
            "intent",
            "general",
        )

        # General assistant.
        if intent == "general":
            return await orchestrator.execute_general(
                state
            )

        # Planning is already complete.
        # PlanningAgent creates plans; it does not execute them.
        if intent == "planning":
            return state

        # Execute specialized agent.
        return await orchestrator.execute_specialized(
            state
        )

    return execute


# ======================================================================
# PLANNING RESULT NORMALIZATION
# ======================================================================


def _normalize_planning_result(
    state: PhoenixState,
    result: Any,
) -> PhoenixState:
    """
    Convert PlanningAgent output into a stable PhoenixState.

    Guarantees:

        response
        plan
        metadata.planning

    And, whenever a structured plan exposes steps:

        metadata.step_count

    Also normalizes:

        metadata.status

    without assuming a specific Plan implementation.
    """

    metadata = dict(
        state.get("metadata") or {}
    )

    content = (
        getattr(
            result,
            "content",
            None,
        )
        or ""
    )

    success = bool(
        getattr(
            result,
            "success",
            False,
        )
    )

    result_data = getattr(
        result,
        "data",
        None,
    )

    if not isinstance(
        result_data,
        dict,
    ):
        result_data = {}

    # Agent metadata has priority over existing metadata.
    metadata.update(
        result_data
    )

    # Planning marker is always present.
    metadata["planning"] = True

    # --------------------------------------------------------------
    # Failure
    # --------------------------------------------------------------

    if not success:

        error = getattr(
            result,
            "error",
            None,
        )

        failed_state = {
            **state,
            "response": content,
            "plan": content,
            "metadata": metadata,
        }

        if error:
            failed_state["error"] = str(
                error
            )

        metadata.setdefault(
            "status",
            "failed",
        )

        return failed_state

    # --------------------------------------------------------------
    # Structured plan
    # --------------------------------------------------------------

    structured_plan = (
        result_data.get("plan")
        or metadata.get("plan")
    )

    # --------------------------------------------------------------
    # Step count
    # --------------------------------------------------------------

    step_count = _extract_step_count(
        structured_plan
    )

    if step_count is None:
        step_count = _valid_int(
            result_data.get(
                "step_count"
            )
        )

    if step_count is None:
        step_count = _valid_int(
            metadata.get(
                "step_count"
            )
        )

    if step_count is not None:
        metadata[
            "step_count"
        ] = step_count

    # --------------------------------------------------------------
    # Status
    # --------------------------------------------------------------

    status = _extract_plan_status(
        structured_plan
    )

    if status:
        metadata[
            "status"
        ] = status
    else:
        metadata.setdefault(
            "status",
            "ready",
        )

    # --------------------------------------------------------------
    # Final state
    # --------------------------------------------------------------

    return {
        **state,
        "response": content,
        "plan": content,
        "metadata": metadata,
    }


# ======================================================================
# PLAN HELPERS
# ======================================================================


def _extract_step_count(
    plan: Any,
) -> int | None:
    """
    Extract step count from supported plan representations.
    """

    if plan is None:
        return None

    if isinstance(
        plan,
        dict,
    ):
        steps = plan.get(
            "steps"
        )

    else:
        steps = getattr(
            plan,
            "steps",
            None,
        )

    if isinstance(
        steps,
        (list, tuple),
    ):
        return len(steps)

    return None


def _extract_plan_status(
    plan: Any,
) -> str | None:
    """
    Extract plan status from dictionaries, dataclasses,
    Pydantic models, or enums.
    """

    if plan is None:
        return None

    if isinstance(
        plan,
        dict,
    ):
        status = plan.get(
            "status"
        )
    else:
        status = getattr(
            plan,
            "status",
            None,
        )

    if status is None:
        return None

    value = getattr(
        status,
        "value",
        status,
    )

    return (
        str(value)
        if value is not None
        else None
    )


def _valid_int(
    value: Any,
) -> int | None:
    """
    Return a non-negative integer or None.
    """

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        int,
    ) and value >= 0:
        return value

    return None


# ======================================================================
# AGENT CONTEXT
# ======================================================================


def _agent_context(
    state: PhoenixState,
) -> AgentContext:
    """
    Build the standard context passed to every Phoenix agent.
    """

    return AgentContext(
        request_id=state[
            "request_id"
        ],
        user_id=state[
            "user_id"
        ],
        project_id=state.get(
            "project_id"
        ),
        metadata=dict(
            state.get(
                "metadata"
            )
            or {}
        ),
        memory_context=list(state.get("memory_context") or []),
    )