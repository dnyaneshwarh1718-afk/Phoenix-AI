from app.agents.base_agent import AgentContext, AgentResult, BaseAgent


class MemoryAgent(BaseAgent):
    name = "memory"

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        return AgentResult(
            success=True,
            content="Memory Agent foundation is ready; persistent memory is scheduled for a later phase.",
        )
