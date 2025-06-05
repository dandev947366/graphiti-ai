from graphiti import EntityNode, EntityEdge, EpisodicNode, EpisodicEdge, AsyncDriver
from datetime import datetime
import asyncio
import uuid
import ollama


class GraphitiCRUD:
    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    async def create_entity_node(self, name: str, summary: str = None):
        """Create a new EntityNode"""
        node = EntityNode(
            uuid=str(uuid.uuid4()),
            name=name,
            labels=["Entity"],
            created_at=datetime.now(),
            summary=summary,
        )
        await node.save(self.driver)
        return node

    async def get_entity_node(self, uuid: str):
        """Retrieve an EntityNode by UUID"""
        return await EntityNode.get_by_uuid(self.driver, uuid)

    async def update_entity_node(self, uuid: str, **kwargs):
        """Update an EntityNode"""
        node = await self.get_entity_node(uuid)
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        await node.save(self.driver)
        return node

    async def delete_entity_node(self, uuid: str):
        """Delete an EntityNode"""
        node = await self.get_entity_node(uuid)
        return await node.delete(self.driver)

    async def create_entity_edge(
        self,
        source_uuid: str,
        target_uuid: str,
        relationship_type: str,
        summary: str = None,
    ):
        """Create a relationship between two EntityNodes"""
        edge = EntityEdge(
            uuid=str(uuid.uuid4()),
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            relationship_type=relationship_type,
            summary=summary,
            created_at=datetime.now(),
        )
        await edge.save(self.driver)
        return edge

    async def get_entity_edge(self, uuid: str):
        """Retrieve an EntityEdge by UUID"""
        return await EntityEdge.get_by_uuid(self.driver, uuid)

    async def delete_entity_edge(self, uuid: str):
        """Delete an EntityEdge"""
        edge = await self.get_entity_edge(uuid)
        return await edge.delete(self.driver)

    # Similar methods for EpisodicNode and EpisodicEdge
    async def create_episodic_node(
        self, name: str, timestamp: datetime, summary: str = None
    ):
        """Create a new EpisodicNode"""
        node = EpisodicNode(
            uuid=str(uuid.uuid4()),
            name=name,
            labels=["Episodic"],
            created_at=datetime.now(),
            timestamp=timestamp,
            summary=summary,
        )
        await node.save(self.driver)
        return node

    async def create_episodic_edge(
        self,
        source_uuid: str,
        target_uuid: str,
        relationship_type: str,
        timestamp: datetime,
        summary: str = None,
    ):
        """Create a relationship between two EpisodicNodes"""
        edge = EpisodicEdge(
            uuid=str(uuid.uuid4()),
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            relationship_type=relationship_type,
            summary=summary,
            created_at=datetime.now(),
            timestamp=timestamp,
        )
        await edge.save(self.driver)
        return edge

    async def generate_embedding(self, text: str):
        """Generate embedding using local Ollama Mistral"""
        response = await ollama.embeddings(model="mistral", prompt=text)
        return response["embedding"]

    async def create_node_with_embedding(self, name: str, summary: str = None):
        """Create node with generated embedding"""
        embedding = await self.generate_embedding(name)
        node = EntityNode(
            uuid=str(uuid.uuid4()),
            name=name,
            name_embedding=embedding,
            labels=["Entity"],
            created_at=datetime.now(),
            summary=summary,
        )
        await node.save(self.driver)
        return node


async def main():
    # Initialize the Neo4j driver (replace with your actual connection details)
    driver = AsyncDriver(uri="bolt://localhost:7687", auth=("neo4j", "password"))

    crud = GraphitiCRUD(driver)

    try:
        # Example CRUD operations
        print("Creating entities...")
        person = await crud.create_entity_node("John Doe", "A person")
        place = await crud.create_entity_node("New York", "A city")

        print("\nCreating relationship...")
        relationship = await crud.create_entity_edge(
            person.uuid, place.uuid, "LIVES_IN", "John lives in New York"
        )

        print("\nRetrieving node...")
        retrieved_person = await crud.get_entity_node(person.uuid)
        print(f"Retrieved: {retrieved_person.name} - {retrieved_person.summary}")

        print("\nUpdating node...")
        updated_person = await crud.update_entity_node(
            person.uuid, summary="Updated summary for John Doe"
        )
        print(f"Updated summary: {updated_person.summary}")

        print("\nCreating episodic data...")
        event = await crud.create_episodic_node(
            "Meeting", datetime(2023, 12, 15, 14, 30), "Team meeting with John"
        )
        event_rel = await crud.create_episodic_edge(
            person.uuid,
            event.uuid,
            "ATTENDED",
            datetime(2023, 12, 15, 14, 30),
            "John attended the meeting",
        )

        # Cleanup (comment out if you want to keep the data)
        print("\nCleaning up...")
        await crud.delete_entity_edge(relationship.uuid)
        await crud.delete_entity_edge(event_rel.uuid)
        await crud.delete_entity_node(place.uuid)
        await crud.delete_entity_node(person.uuid)
        await crud.delete_entity_node(event.uuid)

    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
