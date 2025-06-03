import os
import logging
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatMemory:
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    async def connect(self):
        logger.info("Connecting to Neo4j database...")
        self.driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        logger.info("Connected.")

    async def close(self):
        if self.driver:
            logger.info("Closing Neo4j connection...")
            await self.driver.close()
            logger.info("Connection closed.")

    async def run_query(self, query, parameters=None):
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            records = []
            async for record in result:
                records.append(record.data())
            return records

    async def save_message(self, user_id: str, text: str, sender: str):
        """
        Save a chat message to the graph.
        sender: 'user' or 'bot'
        """
        query = """
            MERGE (u:User {id: $user_id})
            CREATE (m:Message {text: $text, sender: $sender, timestamp: datetime()})
            CREATE (u)-[:SENT]->(m)
        """
        parameters = {"user_id": user_id, "text": text, "sender": sender}
        await self.run_query(query, parameters)
        logger.info(f"Saved message from {sender} for user {user_id}")

    async def get_recent_messages(self, user_id: str, limit: int = 10):
        """
        Retrieve the last `limit` messages for a user, sorted ascending by time.
        """
        query = """
            MATCH (u:User {id: $user_id})-[:SENT]->(m:Message)
            RETURN m.text AS text, m.sender AS sender, m.timestamp AS timestamp
            ORDER BY m.timestamp DESC
            LIMIT $limit
        """
        parameters = {"user_id": user_id, "limit": limit}
        records = await self.run_query(query, parameters)
        # Reverse to chronological order
        return list(reversed(records))

    async def build_prompt(self, user_id: str, new_message: str, limit: int = 10):
        """
        Build a chat prompt including recent history + new message.
        Format adapted to your LLM API input format.
        """
        history = await self.get_recent_messages(user_id, limit)
        prompt = []
        for msg in history:
            role = "user" if msg["sender"] == "user" else "assistant"
            prompt.append({"role": role, "content": msg["text"]})
        # Add new user message last
        prompt.append({"role": "user", "content": new_message})
        return prompt


# Async example usage
import asyncio


async def main():
    URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    USER = os.getenv("NEO4J_USER", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD", "password")  # Use .env for real passwords!

    memory = ChatMemory(URI, USER, PASSWORD)
    await memory.connect()

    user_id = "user123"

    # Save some messages (simulate chat)
    await memory.save_message(user_id, "Hello, how are you?", "user")
    await memory.save_message(user_id, "I am good, thanks for asking!", "bot")
    await memory.save_message(
        user_id, "Can you tell me about renewable energy?", "user"
    )

    # Build prompt for new message
    prompt = await memory.build_prompt(user_id, "What's the latest on solar power?")

    for p in prompt:
        print(f"{p['role']}: {p['content']}")

    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
