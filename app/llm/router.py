from app.core.config import Settings
from app.llm.models import Provider


class ModelRouter:
    """
    Phoenix model routing layer.

    Policy:

    1. Ollama/Qwen is the default.
    2. Gemini is optional.
    3. No OpenAI.
    4. No Anthropic.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def select(
        self,
        task_type: str = "general",
    ) -> tuple[Provider, str]:

        task = task_type.lower().strip()

        # PRIMARY MODEL

        # Phoenix is local-first.
        #
        # All normal operations use Qwen3 through Ollama.

        if task in {
            "general",
            "rag",
            "coding",
            "planning",
            "reasoning",
            "classification",
            "memory",
            "research",
        }:
            return (
                "ollama",
                self.settings.ollama_model,
            )

        # GEMINI
        # Gemini is used only when explicitly routed to it.

        if task in {
            "gemini",
            "cloud_reasoning",
            "advanced_reasoning",
        }:

            if self.settings.gemini_api_key:
                return (
                    "gemini",
                    self.settings.gemini_model,
                )

            # No Gemini key -> safely fall back to local Qwen.
            return (
                "ollama",
                self.settings.ollama_model,
            )

        # FALLBACK
        return (
            "ollama",
            self.settings.ollama_model,
        )