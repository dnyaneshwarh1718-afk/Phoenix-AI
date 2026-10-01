from app.agents.base_agent import AgentContext, AgentResult, BaseAgent


class ApplicationControlAgent(BaseAgent):
    name = "application"

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        return AgentResult(
            success=True,
            content=(
                "Application Control Agent is registered, but desktop application "
                "execution is intentionally disabled in Phase 1. The tool layer and "
                "approval system will be connected before Excel/Word/PowerPoint/Power BI "
                "actions are enabled."
            ),
        )
