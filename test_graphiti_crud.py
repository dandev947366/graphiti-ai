import pytest
import asyncio
from datetime import datetime, timezone
from graphiti.async_driver import AsyncDriver

from crud_graphiti import GraphitiCRUD
import graphiti

print(dir(graphiti))  # to see all top-level attributes
print(dir(graphiti.driver))  # if this import works


@pytest.fixture(scope="module")
async def driver():
    driver = AsyncDriver(uri="bolt://localhost:7687", auth=("neo4j", "password"))
    yield driver
    await driver.close()


@pytest.fixture
async def crud(driver):
    return GraphitiCRUD(driver)


@pytest.mark.asyncio
async def test_entity_node_crud(crud):
    # Create
    node = await crud.create_entity_node("Test Entity", "A test entity")
    assert node.uuid is not None
    assert node.name == "Test Entity"
    assert node.summary == "A test entity"

    # Retrieve
    retrieved = await crud.get_entity_node(node.uuid)
    assert retrieved.uuid == node.uuid
    assert retrieved.name == "Test Entity"

    # Update
    updated = await crud.update_entity_node(node.uuid, summary="Updated summary")
    assert updated.summary == "Updated summary"

    # Delete
    deleted = await crud.delete_entity_node(node.uuid)
    assert deleted is True or deleted is None  # Depends on implementation


@pytest.mark.asyncio
async def test_entity_edge_crud(crud):
    # Setup nodes
    source = await crud.create_entity_node("Source", "Source node")
    target = await crud.create_entity_node("Target", "Target node")

    # Create edge
    edge = await crud.create_entity_edge(
        source.uuid, target.uuid, "RELATED_TO", "Relationship summary"
    )
    assert edge.uuid is not None
    assert edge.relationship_type == "RELATED_TO"

    # Retrieve edge
    retrieved_edge = await crud.get_entity_edge(edge.uuid)
    assert retrieved_edge.uuid == edge.uuid
    assert retrieved_edge.source_uuid == source.uuid

    # Delete edge
    deleted_edge = await crud.delete_entity_edge(edge.uuid)
    assert deleted_edge is True or deleted_edge is None

    # Cleanup nodes
    await crud.delete_entity_node(source.uuid)
    await crud.delete_entity_node(target.uuid)


@pytest.mark.asyncio
async def test_episodic_node_and_edge(crud):
    timestamp = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Create episodic node
    node = await crud.create_episodic_node("Event", timestamp, "An event node")
    assert node.uuid is not None
    assert node.name == "Event"
    assert node.timestamp == timestamp

    # Create episodic edge
    source_node = await crud.create_entity_node("SourceEntity")
    edge = await crud.create_episodic_edge(
        source_node.uuid, node.uuid, "OCCURRED_AT", timestamp, "Edge summary"
    )
    assert edge.uuid is not None
    assert edge.relationship_type == "OCCURRED_AT"
    assert edge.timestamp == timestamp

    # Cleanup
    await crud.delete_entity_edge(edge.uuid)
    await crud.delete_episodic_node(node.uuid)
    await crud.delete_entity_node(source_node.uuid)


@pytest.mark.asyncio
async def test_create_node_with_embedding(crud):
    node = await crud.create_node_with_embedding("Embedded Node", "With embedding")
    assert node.uuid is not None
    assert hasattr(node, "name_embedding")
    # Optionally: assert that name_embedding is a list or vector of floats
