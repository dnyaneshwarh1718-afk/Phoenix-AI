from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.core.logger import get_logger

from app.llm.models import ModelRequest, ModelResponse
from app.llm.router import ModelRouter


logger = get_logger(__name__)


class LLMGateway:
    """
    Phoenix unified LLM gateway.

    Supported providers:

        - Ollama
        - Gemini

    Ollama is the default provider.
    """

    def __init__(self, settings: Settings):

        self.settings = settings
        self.router = ModelRouter(settings)

    # PUBLIC API

    async def chat(
        self,
        request: ModelRequest,
    ) -> ModelResponse:

        provider, model = self._resolve(request)

        logger.info(
            "LLM request | provider=%s | model=%s | task=%s",
            provider,
            model,
            request.task_type,
        )

        if provider == "ollama":
            return await self._ollama_chat(
                request=request,
                model=model,
            )

        if provider == "gemini":
            return await self._gemini_chat(
                request=request,
                model=model,
            )

        raise LLMError(
            f"Unsupported LLM provider: {provider}"
        )

    # ROUTING

    def _resolve(
        self,
        request: ModelRequest,
    ) -> tuple[str, str]:

        # Explicit provider/model requested.
        if request.provider:

            if request.model:
                return (
                    request.provider,
                    request.model,
                )

            if request.provider == "ollama":
                return (
                    "ollama",
                    self.settings.ollama_model,
                )

            if request.provider == "gemini":
                return (
                    "gemini",
                    self.settings.gemini_model,
                )

        # Otherwise use Phoenix router.
        return self.router.select(
            request.task_type
        )

    # OLLAMA

    async def _ollama_chat(
        self,
        request: ModelRequest,
        model: str,
    ) -> ModelResponse:

        url = (
            f"{self.settings.ollama_base_url}"
            "/api/chat"
        )

        payload = {
            "model": model,
            "messages": request.messages,
            "stream": False,
            "options": {
                "temperature": (
                    self.settings.temperature
                    if request.temperature is None
                    else request.temperature
                )
            },
        }

        # Ollama supports provider-side JSON mode. This is stronger than
        # prompt-only JSON instructions and prevents the planning agent from
        # receiving prose/fenced output that cannot be validated.
        if request.json_mode:
            payload["format"] = "json"

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        try:

            async with httpx.AsyncClient(
                timeout=300
            ) as client:

                response = await client.post(
                    url,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

            message = data.get(
                "message",
                {},
            )

            content = message.get(
                "content",
                "",
            )

            usage = {
                "prompt_tokens": data.get(
                    "prompt_eval_count"
                ),
                "completion_tokens": data.get(
                    "eval_count"
                ),
                "total_duration_ns": data.get(
                    "total_duration"
                ),
            }

            return ModelResponse(
                content=content,
                provider="ollama",
                model=model,
                usage=usage,
                raw=data,
            )

        except Exception as exc:

            logger.exception(
                "Ollama request failed"
            )

            raise LLMError(
                f"Ollama call failed: "
                f"{model}: {exc}"
            ) from exc

    # GEMINI

    async def _gemini_chat(
        self,
        request: ModelRequest,
        model: str,
    ) -> ModelResponse:

        api_key = (
            self.settings.gemini_api_key
        )

        if not api_key:

            raise LLMError(
                "Gemini API key is not configured."
            )

        # Convert Phoenix messages into Gemini format.
        contents = []

        system_instruction = None

        for message in request.messages:

            role = message["role"]
            content = message["content"]

            if role == "system":

                system_instruction = {
                    "parts": [
                        {
                            "text": content
                        }
                    ]
                }

            else:

                gemini_role = (
                    "model"
                    if role == "assistant"
                    else "user"
                )

                contents.append(
                    {
                        "role": gemini_role,
                        "parts": [
                            {
                                "text": content
                            }
                        ],
                    }
                )

        url = (
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model}:generateContent"
            f"?key={api_key}"
        )

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": (
                    self.settings.temperature
                    if request.temperature is None
                    else request.temperature
                ),
                "maxOutputTokens": (
                    self.settings.max_tokens
                    if request.max_tokens is None
                    else request.max_tokens
                ),
            },
        }

        if system_instruction:

            payload[
                "systemInstruction"
            ] = system_instruction

        if request.json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        try:

            async with httpx.AsyncClient(
                timeout=180
            ) as client:

                response = await client.post(
                    url,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

            candidates = data.get(
                "candidates",
                [],
            )

            if not candidates:

                raise LLMError(
                    f"Gemini returned no candidates: "
                    f"{data}"
                )

            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )

            text_parts = []

            for part in parts:

                text = part.get(
                    "text"
                )

                if text:
                    text_parts.append(text)

            content = "\n".join(
                text_parts
            )

            usage = data.get(
                "usageMetadata",
                {},
            )

            return ModelResponse(
                content=content,
                provider="gemini",
                model=model,
                usage=usage,
                raw=data,
            )

        except httpx.HTTPStatusError as exc:

            body = exc.response.text

            logger.error(
                "Gemini API error: %s",
                body,
            )

            raise LLMError(
                f"Gemini API failed "
                f"({exc.response.status_code}): "
                f"{body}"
            ) from exc

        except Exception as exc:

            logger.exception(
                "Gemini request failed"
            )

            raise LLMError(
                f"Gemini call failed: "
                f"{model}: {exc}"
            ) from exc