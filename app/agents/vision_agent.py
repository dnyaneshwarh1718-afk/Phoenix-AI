from app.agents.base_agent import AgentContext, AgentResult, BaseAgent


class ComputerVisionAgent(BaseAgent):
    name = "vision"

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        return AgentResult(
            success=True,
            content="Computer Vision Agent foundation is registered; GUI vision is scheduled for a later phase.",
        )
