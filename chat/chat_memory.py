import asyncio
import json
import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

from neo4j import AsyncGraphDatabase
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")


class ChatMemory:
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.graphiti = None

    async def save_message(self, user_id: str, text: str, sender: str):
        await self.save_message_as_episode(user_id, text, sender)

    async def run_query(self, query: str, params: dict):
        async with self.driver.session() as session:
            result = await session.run(query, params)
            records = []
            async for record in result:
                records.append(record.data())
            return records

    async def connect(self):
        logger.info("Connecting to Neo4j...")
        self.driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        logger.info("Neo4j connected.")
        self.graphiti = Graphiti(self.uri, self.user, self.password)
        logger.info("Initializing Graphiti indices and constraints...")
        await self.graphiti.build_indices_and_constraints()
        logger.info("Graphiti ready.")

    async def close(self):
        if self.driver:
            logger.info("Closing Neo4j driver...")
            await self.driver.close()
            logger.info("Neo4j driver closed.")
        if self.graphiti:
            await self.graphiti.close()
            logger.info("Graphiti connection closed.")

    async def save_message_as_episode(self, user_id: str, text: str, sender: str):
        """
        Save chat message both as a Neo4j Message node and as a Graphiti Episode.
        """
        # Save basic chat message node
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (u:User {id: $user_id})
                CREATE (m:Message {text: $text, sender: $sender, timestamp: datetime()})
                CREATE (u)-[:SENT]->(m)
                """,
                {"user_id": user_id, "text": text, "sender": sender},
            )
            logger.info(f"Saved message from {sender} for user {user_id}")

        # Add message as an episode for semantic search (using Graphiti)
        # Use 'text' EpisodeType for chat messages
        await self.graphiti.add_episode(
            name=f"Chat message from {sender} by {user_id} at {datetime.now(timezone.utc).isoformat()}",
            episode_body=text,
            source=EpisodeType.text,
            source_description="chat message",
            reference_time=datetime.now(timezone.utc),
        )
        logger.info("Saved message as Graphiti episode.")

    async def search_knowledge(self, query: str, limit: int = 5):
        """
        Use Graphiti semantic + BM25 hybrid search to find relevant facts.
        """
        results = await self.graphiti.search(query)
        limited_results = results[:limit]
        logger.info(f"Found {len(limited_results)} search results for query: {query}")
        return limited_results

    async def get_recent_messages(self, user_id: str, limit: int = 10):
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (u:User {id: $user_id})-[:SENT]->(m:Message)
                RETURN m.text AS text, m.sender AS sender, m.timestamp AS timestamp
                ORDER BY m.timestamp DESC
                LIMIT $limit
                """,
                {"user_id": user_id, "limit": limit},
            )
            records = []
            async for record in result:
                records.append(record.data())
            return list(reversed(records))

    async def build_prompt(self, user_id: str, new_message: str, limit: int = 10):
        history = await self.get_recent_messages(user_id, limit)
        prompt = []
        for msg in history:
            role = "user" if msg["sender"] == "user" else "assistant"
            prompt.append({"role": role, "content": msg["text"]})
        prompt.append({"role": "user", "content": new_message})
        return prompt


async def main():
    chat = ChatMemory(neo4j_uri, neo4j_user, neo4j_password)
    await chat.connect()

    user_id = "user123"

    # Save messages as both Neo4j nodes and Graphiti episodes
    await chat.save_message_as_episode(user_id, "Hello, how are you?", "user")
    await chat.save_message_as_episode(user_id, "I am good, thanks!", "bot")
    await chat.save_message_as_episode(user_id, "Tell me about Kamala Harris.", "user")

    # Search Graphiti knowledge base semantically
    results = await chat.search_knowledge("Who is Kamala Harris?")
    for res in results:
        print(f"Fact: {res.fact}")

    await chat.close()


if __name__ == "__main__":
    asyncio.run(main())
