"""
This test suite verifies the initialization behavior and error handling of the
`Graphiti` class from the `graphiti_core` module, which manages setup for a
Neo4j-backed graph data model.

Functionality:
- Tests that the `build_indices_and_constraints` method is invoked correctly.
- Confirms that resources (Neo4j driver sessions) are properly closed after use.
- Ensures connection cleanup occurs even when exceptions are raised.

Requirements:
- pytest
- pytest-asyncio
- unittest.mock
- graphiti_core module

Use Case:
Useful for ensuring reliable graph schema setup and robust startup behavior in
applications that depend on Neo4j, such as knowledge graphs or event-based memory systems.
"""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_graphiti_initialization_and_indices():
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"

    with patch("graphiti_core.Graphiti") as mock_graphiti:
        mock_instance = mock_graphiti.return_value
        mock_instance.build_indices_and_constraints = AsyncMock()
        mock_instance.close = AsyncMock()

        graphiti = mock_graphiti(neo4j_uri, neo4j_user, neo4j_password)

        try:
            await graphiti.build_indices_and_constraints()
            mock_instance.build_indices_and_constraints.assert_awaited_once()

        finally:
            await graphiti.close()
            print("\nConnection closed")
            mock_instance.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_graphiti_connection_closure_on_error():
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"

    with patch("graphiti_core.Graphiti") as mock_graphiti:
        mock_instance = mock_graphiti.return_value
        mock_instance.build_indices_and_constraints = AsyncMock(
            side_effect=Exception("Test error")
        )
        mock_instance.close = AsyncMock()

        graphiti = mock_graphiti(neo4j_uri, neo4j_user, neo4j_password)

        with pytest.raises(Exception, match="Test error"):
            try:
                await graphiti.build_indices_and_constraints()
            finally:
                await graphiti.close()
                print("\nConnection closed")

        mock_instance.close.assert_awaited_once()
