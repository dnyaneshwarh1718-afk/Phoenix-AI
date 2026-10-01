class PhoenixError(Exception):
    """Base Phoenix exception."""


class LLMError(PhoenixError):
    """Raised when an LLM provider fails."""


class AgentError(PhoenixError):
    """Raised when an agent cannot complete its operation."""


class ToolError(PhoenixError):
    """Raised when a tool fails."""


class ApprovalRequired(PhoenixError):
    """Raised when a tool requires explicit human approval."""
