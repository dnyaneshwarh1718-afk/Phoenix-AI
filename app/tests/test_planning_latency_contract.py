from app.llm.gateway import LLMGateway
from app.llm.models import ModelRequest


def test_reasoning_requests_disable_qwen_thinking_and_keep_json(monkeypatch):
    class Settings:
        ollama_base_url = "http://127.0.0.1:11434"
        ollama_model = "qwen3:4b-instruct"
        temperature = 0.1

    gateway = LLMGateway.__new__(LLMGateway)
    gateway.settings = Settings()

    captured = {}
    async def fake_post(*args, **kwargs):
        captured.update(kwargs.get("json", {}))
        class R:
            is_error = False
            status_code = 200
            text = ""
            def raise_for_status(self): pass
            def json(self): return {"message": {"content": "{}"}}
        return R()

    # Structural contract test: verify the payload-building behavior by
    # monkeypatching the AsyncClient context manager.
    class Client:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def post(self, *args, **kwargs): return await fake_post(*args, **kwargs)

    monkeypatch.setattr("app.llm.gateway.httpx.AsyncClient", Client)
    import asyncio
    asyncio.run(gateway._ollama_chat(ModelRequest(
        messages=[{"role": "user", "content": "plan"}],
        task_type="reasoning", json_mode=True, max_tokens=768,
    ), "qwen3:4b-instruct"))
    assert captured["think"] is False
    assert captured["format"] == "json"
    assert captured["options"]["num_predict"] == 768
