# Phoenix AI Phase 0/1 Architecture

## Design rules

1. Agents own reasoning and task-specific behavior.
2. Tools own external actions.
3. Orchestrator owns coordination.
4. LLM Gateway owns model/provider abstraction.
5. LangGraph owns workflow/state transitions.
6. Security/approval sits between intent and dangerous execution.
7. Future MCP integrations must plug into the tool registry rather than bypassing the agent architecture.

## Current agents

- Planning
- RAG foundation
- Research foundation
- Application Control foundation
- Memory foundation
- Computer Vision foundation

## Current limitations

No real filesystem automation, web search, desktop automation, document indexing, persistent memory, or MCP execution is enabled yet. Those are deliberately isolated for later phases.
