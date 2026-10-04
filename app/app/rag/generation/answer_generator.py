from dataclasses import dataclass

import requests


@dataclass
class AnswerGenerationResult:
    """
    Standard result returned by the answer generator.
    """

    answer: str
    model: str
    provider: str = "ollama"


class AnswerGenerator:
    """
    Generates grounded answers using a local LLM through Ollama.

    Architecture:

        Prompt
          ↓
        Ollama
          ↓
        Qwen3
          ↓
        AnswerGenerationResult
    """

    def __init__(
        self,
        model_name: str = "qwen3:4b-instruct",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120,
    ) -> None:

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> AnswerGenerationResult:

        if not messages:
            raise ValueError(
                "Messages cannot be empty."
            )

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            },
        }

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Failed to communicate with Ollama: {exc}"
            ) from exc

        data = response.json()

        message = data.get(
            "message",
            {},
        )

        answer = message.get(
            "content",
            "",
        )

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        return AnswerGenerationResult(
            answer=answer.strip(),
            model=self.model_name,
            provider="ollama",
        )

    def generate_from_context(
        self,
        query: str,
        context: str,
        prompt_builder,
    ) -> AnswerGenerationResult:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not context or not context.strip():
            raise ValueError(
                "Context cannot be empty."
            )

        messages = prompt_builder.build(
            query=query.strip(),
            context=context.strip(),
        )

        return self.generate(messages)