"""
This module contains asynchronous pytest tests for the ChatMemory class that manages
chat messages stored in a Neo4j database.

Functionality:
- Sets up an async fixture to instantiate and connect a ChatMemory instance using Neo4j
  connection parameters loaded from environment variables.
- Tests saving messages to Neo4j and verifies that user nodes and message nodes are
  created and retrievable.
- Checks retrieval of recent messages for a user, ensuring correct order and content.
- Verifies the prompt-building function correctly compiles chat history and appends new
  messages.

Requirements:
- Neo4j instance running and accessible via credentials in environment variables:
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
- pytest and pytest-asyncio installed for async test support.
- ChatMemory class implemented with async methods: connect, close, save_message,
  get_recent_messages, and build_prompt.

Use Case:
Used to verify correct storage, retrieval, and formatting of chat conversation data in
applications that use Neo4j to persist chat histories, enabling conversational AI or
chatbot memory management.
"""

import os
import pytest
import pytest_asyncio
from chat.chat_memory import (
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
