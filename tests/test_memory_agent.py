import pytest

from app.agents.base_agent import AgentContext
from app.agents.memory_agent import MemoryAgent
from app.memory.store import MemoryStore


def make_agent(tmp_path):
    return MemoryAgent(MemoryStore(str(tmp_path / "memory.db")))


@pytest.mark.asyncio
async def test_memory_agent_remember_and_recall(tmp_path):
    agent = make_agent(tmp_path)
    context = AgentContext("req-1", user_id="alice")

    stored = await agent.run("Remember that my main project is Phoenix AI", context)
    assert stored.success
    assert stored.data["status"] == "stored"

    recalled = await agent.run("What do you remember about Phoenix AI?", context)
    assert recalled.success
    assert "main project is Phoenix AI" in recalled.content
    assert recalled.data["memory_count"] >= 1


@pytest.mark.asyncio
async def test_memory_is_user_scoped(tmp_path):
    agent = make_agent(tmp_path)
    await agent.run("Remember that my preferred editor is VS Code", AgentContext("r1", user_id="alice"))

    other = await agent.run("What do you remember about editor?", AgentContext("r2", user_id="bob"))
    assert other.success
    assert other.data["memory_count"] == 0


@pytest.mark.asyncio
async def test_memory_persists_across_agent_instances(tmp_path):
    db = str(tmp_path / "memory.db")
    first = MemoryAgent(MemoryStore(db))
    await first.run("Remember that the deployment target is local Windows", AgentContext("r1", user_id="alice"))

    second = MemoryAgent(MemoryStore(db))
    result = await second.run("Do you remember the deployment target?", AgentContext("r2", user_id="alice"))
    assert result.success
    assert "local Windows" in result.content


@pytest.mark.asyncio
async def test_memory_forget_deactivates_matching_memory(tmp_path):
    agent = make_agent(tmp_path)
    context = AgentContext("r1", user_id="alice")
    await agent.run("Remember that the demo database is SQLite", context)

    forgotten = await agent.run("Forget the demo database", context)
    assert forgotten.success
    assert forgotten.data["forgotten_count"] == 1

    recalled = await agent.run("What do you remember about the demo database?", context)
    assert recalled.data["memory_count"] == 0


def test_store_deduplicates(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    a = store.add(user_id="alice", content="Phoenix AI is local-first")
    b = store.add(user_id="alice", content="Phoenix AI is local-first")
    assert a.memory_id == b.memory_id
    assert store.count(user_id="alice") == 1


def test_store_captures_conversation_turn(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    records = store.save_turn(
        user_id="alice",
        user_message="We are building Phoenix AI",
        assistant_response="Phoenix AI is the current project.",
    )
    assert len(records) == 2
    matches = store.search(user_id="alice", query="Phoenix AI", limit=5)
    assert len(matches) == 2


def test_store_supports_natural_language_retrieval_and_in_memory_database():
    store = MemoryStore(":memory:")
    store.add(user_id="alice", content="Phoenix AI uses Qdrant", kind="fact")

    matches = store.search(
        user_id="alice",
        query="What vector database does Phoenix AI use?",
        limit=8,
    )

    assert matches
    assert matches[0].content == "Phoenix AI uses Qdrant"
