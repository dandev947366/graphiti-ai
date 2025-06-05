import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import uuid
import numpy as np
from graphiti import EntityNode, EntityEdge, EpisodicNode, EpisodicEdge, AsyncDriver
from crud_graphiti import GraphitiCRUD


@pytest.fixture
def mock_driver():
    driver = MagicMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock()
    return driver


@pytest.fixture
def crud(mock_driver):
    return GraphitiCRUD(mock_driver)


@pytest.fixture
def sample_entity_node():
    return EntityNode(
        uuid=str(uuid.uuid4()),
        name="Test Node",
        labels=["Entity"],
        created_at=datetime.now(),
        summary="Test summary",
    )


@pytest.fixture
def sample_episodic_node():
    return EpisodicNode(
        uuid=str(uuid.uuid4()),
        name="Episodic Node",
        labels=["Episodic"],
        created_at=datetime.now(),
        timestamp=datetime.now(),
        content="Test content",
    )


@pytest.fixture
def sample_entity_edge(sample_entity_node):
    return EntityEdge(
        uuid=str(uuid.uuid4()),
        source_uuid=sample_entity_node.uuid,
        target_uuid=str(uuid.uuid4()),
        relationship_type="RELATES_TO",
        created_at=datetime.now(),
    )


@pytest.mark.asyncio
async def test_create_entity_node(crud, mock_driver):
    mock_driver.execute_query.return_value = [{"uuid": "test-uuid"}]

    node = await crud.create_entity_node("Test Node", "Test summary")

    assert isinstance(node, EntityNode)
    assert node.name == "Test Node"
    mock_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_entity_node(crud, mock_driver, sample_entity_node):
    mock_driver.execute_query.return_value = [
        {
            "uuid": sample_entity_node.uuid,
            "name": sample_entity_node.name,
            "created_at": sample_entity_node.created_at,
            "summary": sample_entity_node.summary,
        }
    ]

    result = await crud.get_entity_node(sample_entity_node.uuid)

    assert result.uuid == sample_entity_node.uuid
    mock_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_entity_node(crud, mock_driver, sample_entity_node):
    # Mock get
    mock_driver.execute_query.side_effect = [
        [
            {
                "uuid": sample_entity_node.uuid,
                "name": sample_entity_node.name,
                "created_at": sample_entity_node.created_at,
                "summary": sample_entity_node.summary,
            }
        ],
        [{"uuid": sample_entity_node.uuid}],  # Save response
    ]

    updated = await crud.update_entity_node(
        sample_entity_node.uuid, summary="Updated summary"
    )

    assert updated.summary == "Updated summary"
    assert mock_driver.execute_query.await_count == 2


@pytest.mark.asyncio
async def test_delete_entity_node(crud, mock_driver, sample_entity_node):
    # Mock get
    mock_driver.execute_query.side_effect = [
        [
            {
                "uuid": sample_entity_node.uuid,
                "name": sample_entity_node.name,
                "created_at": sample_entity_node.created_at,
                "summary": sample_entity_node.summary,
            }
        ],
        True,  # Delete response
    ]

    result = await crud.delete_entity_node(sample_entity_node.uuid)

    assert result is True
    mock_driver.execute_query.assert_awaited_with(
        "MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n", uuid=sample_entity_node.uuid
    )


@pytest.mark.asyncio
async def test_create_entity_edge(crud, mock_driver, sample_entity_node):
    target_uuid = str(uuid.uuid4())
    mock_driver.execute_query.side_effect = [
        [{"uuid": sample_entity_node.uuid}],  # Source exists
        [{"uuid": target_uuid}],  # Target exists
        [{"uuid": "edge-uuid"}],  # Edge created
    ]

    edge = await crud.create_entity_edge(
        sample_entity_node.uuid, target_uuid, "TEST_RELATIONSHIP"
    )

    assert isinstance(edge, EntityEdge)
    assert edge.relationship_type == "TEST_RELATIONSHIP"
    assert mock_driver.execute_query.await_count == 3


@pytest.mark.asyncio
async def test_create_node_with_embedding(crud, mock_driver):
    mock_embedding = np.array([0.1, 0.2, 0.3])

    with patch.object(
        crud, "generate_embedding", AsyncMock(return_value=mock_embedding)
    ):
        mock_driver.execute_query.return_value = [{"uuid": "test-uuid"}]

        node = await crud.create_node_with_embedding("Test Node")

        assert node.name_embedding.tolist() == mock_embedding.tolist()
        crud.generate_embedding.assert_awaited_once_with("Test Node")


@pytest.mark.asyncio
async def test_transaction(crud, mock_driver):
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_session.begin_transaction.return_value.__aenter__.return_value = mock_tx

    async with crud.transaction() as tx:
        assert tx == mock_tx

    mock_driver.session.assert_called_once()
    mock_session.begin_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling(crud, mock_driver):
    mock_driver.execute_query.side_effect = Exception("DB error")

    with pytest.raises(RuntimeError, match="Failed to create entity node"):
        await crud.create_entity_node("Fail Node")


# Similar tests for EpisodicNode and EpisodicEdge operations...
@pytest.mark.asyncio
async def test_create_episodic_node(crud, mock_driver):
    mock_driver.execute_query.return_value = [{"uuid": "episodic-uuid"}]
    timestamp = datetime.now()

    node = await crud.create_episodic_node("Event", timestamp, "Test event")

    assert isinstance(node, EpisodicNode)
    assert node.timestamp == timestamp
    mock_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_episodic_edge(crud, mock_driver, sample_episodic_node):
    target_uuid = str(uuid.uuid4())
    timestamp = datetime.now()
    mock_driver.execute_query.side_effect = [
        [{"uuid": sample_episodic_node.uuid}],  # Source exists
        [{"uuid": target_uuid}],  # Target exists
        [{"uuid": "edge-uuid"}],  # Edge created
    ]

    edge = await crud.create_episodic_edge(
        sample_episodic_node.uuid, target_uuid, "TEST_RELATIONSHIP", timestamp
    )

    assert isinstance(edge, EpisodicEdge)
    assert edge.timestamp == timestamp
    assert mock_driver.execute_query.await_count == 3
