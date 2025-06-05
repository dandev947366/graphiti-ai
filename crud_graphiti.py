import uuid
import datetime
import asyncio
import logging
from graphiti.graph.node import EntityNode  # Updated import path
from graphiti.driver.neo4j_driver import AsyncNeo4jDriver  # Updated import path
from mistral_client import LocalAiClient  # Local Mistral client
from graphiti_core.prompts import Message
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Neo4j driver
neo4j_driver = AsyncNeo4jDriver(
    uri="bolt://localhost:7687", username="neo4j", password="test"
)

# Setup Mistral client
llm_client = LocalAiClient(base_url="http://localhost:11434")


async def create_entity(name: str, summary: str):
    embedding = llm_client.generate_embedding(
        name
    )  # Assuming this is synchronous; adjust if async
    entity = EntityNode(
        uuid=str(uuid.uuid4()),
        name=name,
        summary=summary,
        name_embedding=embedding,
        created_at=datetime.datetime.utcnow(),
        labels=["Entity"],
    )
    await entity.save(neo4j_driver)
    return entity


async def get_entity_by_uuid(entity_uuid: str):
    return await EntityNode.get_by_uuid(neo4j_driver, entity_uuid)


async def delete_entity(entity: EntityNode):
    await entity.delete(neo4j_driver)


async def main():
    logger.info("Creating entity...")
    new_entity = await create_entity("ChatGPT", "An AI language model by OpenAI.")

    logger.info("Retrieving entity...")
    fetched_entity = await get_entity_by_uuid(new_entity.uuid)
    print("Fetched entity:", fetched_entity)

    logger.info("Deleting entity...")
    await delete_entity(fetched_entity)


if __name__ == "__main__":
    asyncio.run(main())
