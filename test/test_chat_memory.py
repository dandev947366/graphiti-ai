import os
import pytest
import pytest_asyncio
from chat_memory import (
    ChatMemory,
)


@pytest_asyncio.fixture
async def memory():
    URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    USER = os.getenv("NEO4J_USER", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    mem = ChatMemory(URI, USER, PASSWORD)
    await mem.connect()
    yield mem
    await mem.close()


@pytest.mark.asyncio
async def test_save_message_creates_user_node(memory):
    user_id = "testuser3"
    await memory.save_message(user_id, "Testing user creation.", "user")

    messages = await memory.get_recent_messages(user_id)
    assert any(msg["text"] == "Testing user creation." for msg in messages)


@pytest.mark.asyncio
async def test_save_and_retrieve_messages(memory):
    user_id = "testuser1"
    await memory.save_message(user_id, "Hello world", "user")
    await memory.save_message(user_id, "Hi there!", "bot")

    messages = await memory.get_recent_messages(user_id, limit=2)
    assert len(messages) == 2
    assert messages[0]["text"] == "Hello world"
    assert messages[1]["text"] == "Hi there!"


@pytest.mark.asyncio
async def test_build_prompt_includes_history_and_new_message(memory):
    user_id = "testuser2"
    await memory.save_message(user_id, "First message", "user")
    await memory.save_message(user_id, "Second message", "bot")

    new_message = "What about the weather?"
    prompt = await memory.build_prompt(user_id, new_message)

    # The prompt should include history messages plus the new message
    assert prompt[-1]["content"] == new_message
    assert any(p["content"] == "First message" for p in prompt)
    assert any(p["content"] == "Second message" for p in prompt)
