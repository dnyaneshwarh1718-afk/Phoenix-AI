# ADR-001: Phoenix modular agent architecture

## Decision

Phoenix uses a modular architecture with a central Orchestrator, specialized Agents, a provider-neutral LLM Gateway, LangGraph workflows, and a Tool Registry.

## Why

This prevents provider lock-in, keeps agent reasoning separate from external actions, enables approval gates, and allows MCP to be introduced without rewriting the core.

## Rejected approach

A single autonomous LangChain agent controlling every capability directly.

## Consequences

There are more interfaces and files, but the system is substantially easier to test, secure, evolve, and deploy.
