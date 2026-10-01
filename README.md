# Phoenix AI — Phase 0 + Phase 1

A production-oriented foundation for a multipurpose agentic AI system.

## Included

- FastAPI API
- CLI chat interface
- Configuration management with Pydantic Settings
- Structured logging
- Pluggable LLM Gateway using LiteLLM
- OpenAI / Anthropic / Gemini provider support
- Model routing
- Base Agent abstraction
- Phoenix Orchestrator
- LangGraph workflow
- Typed conversation state
- Tool registry foundation
- Safety/approval foundation
- Health endpoint
- Unit tests
- Docker foundation

## Quick start

### 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure `.env`

At minimum configure one provider:

```env
OPENAI_API_KEY=your_key
```

The gateway can also use:

```env
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key
```

### 3. Run API

```bash
uvicorn app.api.main:app --reload
```

Open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

### 4. Run CLI

```bash
python -m app.cli
```

## Example API request

```json
POST /api/v1/chat
{
  "message": "Explain what Phoenix AI is"
}
```

## Architecture

```text
User
  |
  v
FastAPI / CLI
  |
  v
Phoenix Orchestrator
  |
  +--> Planning / routing
  |
  +--> Agent registry
  |
  +--> Tool registry
  |
  +--> LLM Gateway
  |
  v
Response
```

## Next phases

Phase 2: advanced hybrid RAG
Phase 3: real tools and filesystem
Phase 4: application automation
Phase 5: MCP
Phase 6: vision
Phase 7: persistent memory
Phase 8: autonomous execution
Phase 9: evaluation/observability
Phase 10: production deployment
