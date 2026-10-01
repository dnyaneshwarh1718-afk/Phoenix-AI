from langgraph.graph import END, START, StateGraph

from app.orchestrator.state import PhoenixState


def build_phoenix_graph(orchestrator):
    graph = StateGraph(PhoenixState)

    async def classify(state: PhoenixState):
        state["intent"] = orchestrator.router.classify(
            state["user_message"],
            state.get("metadata", {}),
        )
        return state

    def route(state: PhoenixState):
        intent = state["intent"]
        state["selected_agent"] = intent
        return state

    async def plan(state: PhoenixState):
        # Planning is deliberately explicit for requests that benefit from planning.
        if state["intent"] == "planning":
            result = await orchestrator.agents["planning"].run(
                state["user_message"],
                orchestrator_context(state),
            )
            state["plan"] = result.content
            state["metadata"].update(result.data)
            state["response"] = result.content
            if not result.success:
                state["error"] = result.error
            return state

        # RAG requests should go directly to retrieval. Planning here adds latency
        # and can contaminate a grounded workflow with an unnecessary LLM call.
        if state["intent"] == "rag":
            return state

        # Complex requests can get a lightweight plan before execution.
        complex_markers = [
            " and ",
            " then ",
            "after that",
            "create",
            "build",
            "analyze",
        ]
        if any(marker in state["user_message"].lower() for marker in complex_markers):
            result = await orchestrator.agents["planning"].run(
                state["user_message"],
                orchestrator_context(state),
            )
            state["plan"] = result.content
            state["metadata"].update(result.data)
            if not result.success:
                state["error"] = result.error
        return state

    async def execute(state: PhoenixState):
        if state["intent"] in {"general", "planning"}:
            if state["intent"] == "general":
                return await orchestrator.execute_general(state)
            return state
        return await orchestrator.execute_specialized(state)

    graph.add_node("classify", classify)
    graph.add_node("plan", plan)
    graph.add_node("route", route)
    graph.add_node("execute", execute)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "route")
    graph.add_edge("route", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", END)

    return graph.compile()


def orchestrator_context(state: PhoenixState):
    from app.agents.base_agent import AgentContext

    return AgentContext(
        request_id=state["request_id"],
        user_id=state["user_id"],
        project_id=state.get("project_id"),
        metadata=state.get("metadata", {}),
    )
