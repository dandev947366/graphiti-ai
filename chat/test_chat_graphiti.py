import pytest
import pytest_asyncio
from .chat_memory import ChatMemory


@pytest_asyncio.fixture
async def chat_memory():
    cm = ChatMemory("bolt://localhost:7687", "neo4j", "password")
    await cm.connect()
    yield cm
    await cm.close()


@pytest.mark.asyncio
async def test_save_and_get_messages(chat_memory):
    user_id = "testuser"
    await chat_memory.save_message(user_id, "Hello, bot!", "user")
    await chat_memory.save_message(user_id, "Hello, user!", "bot")

    # Retrieve messages and verify they were saved correctly
    messages = await chat_memory.get_recent_messages(user_id, limit=2)
    assert len(messages) == 2
    assert messages[0]["text"] == "Hello, bot!"
    assert messages[0]["sender"] == "user"
    assert messages[1]["text"] == "Hello, user!"
    assert messages[1]["sender"] == "bot"


@pytest.mark.asyncio
async def test_build_prompt(chat_memory):
    user_id = "testuser2"
    await chat_memory.save_message(user_id, "Hi there!", "user")
    await chat_memory.save_message(user_id, "Hello!", "bot")

    prompt = await chat_memory.build_prompt(user_id, "New question?")

    assert prompt[-1]["role"] == "user"
    assert prompt[-1]["content"] == "New question?"
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"] == "Hi there!"
    assert prompt[1]["role"] == "assistant"
    assert prompt[1]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_run_query_mocked():
    memory = ChatMemory("bolt://fake-uri", "user", "pass")

    mock_record = {"text": "mocked", "sender": "user"}

    class MockResult:
        def __aiter__(self):
            self._iter = iter([mock_record])
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def run(self, query, params):
            return MockResult()

    class MockDriver:
        def session(self):
            return MockSession()

        async def close(self):
            pass

    memory.driver = MockDriver()

    # Adjust method name if your actual private query runner is different
    records = await memory._run_query("MATCH (n) RETURN n", {})

    assert len(records) == 1
    assert records[0]["text"] == "mocked"
    assert records[0]["sender"] == "user"
