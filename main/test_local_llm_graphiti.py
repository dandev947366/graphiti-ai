import os
import sys
import pytest
import asyncio
from datetime import datetime

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.local_llm_graphiti import doSearch
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from main.llm_embedder import HuggingFaceEmbedder
from main.local_ai_client import LocalAiClient


@pytest.fixture
async def graphiti_instance():
    llm_config = LLMConfig(
        base_url="http://127.0.0.1:11434/api/generate", temperature=0.5
    )
    local_llm_client = LocalAiClient(config=llm_config, grammar_file="./json.gbnf")
    neo4j_db_name = "neo4j"
    neo4j_db_pass = "password"

    graphiti = Graphiti(
        "bolt://localhost:7687",
        neo4j_db_name,
        neo4j_db_pass,
        embedder=HuggingFaceEmbedder(),
        llm_client=local_llm_client,
    )

    await graphiti.build_indices_and_constraints()
    yield graphiti
    await graphiti.close()


@pytest.mark.asyncio
async def test_add_episode(graphiti_instance):
    episode = "Kamala Harris is the Attorney General of California. She was previously the district attorney for San Francisco."

    result = await graphiti_instance.add_episode(
        name="Test Episode",
        episode_body=episode,
        source=EpisodeType.text,
        source_description="test",
        reference_time=datetime.now(),
    )

    assert result is not None


@pytest.mark.asyncio
async def test_search_functionality(graphiti_instance):
    # First add some test data
    episodes = [
        "Kamala Harris is the Attorney General of California. She was previously the district attorney for San Francisco.",
        "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
    ]

    for i, episode in enumerate(episodes):
        await graphiti_instance.add_episode(
            name=f"Test Episode {i}",
            episode_body=episode,
            source=EpisodeType.text,
            source_description="test",
            reference_time=datetime.now(),
        )

    # Test search
    results = await graphiti_instance.search("Who was the California Attorney General?")
    assert results is not None
    assert len(results) > 0


@pytest.mark.asyncio
async def test_full_do_search():
    # Test the complete doSearch function
    try:
        await doSearch()
    except Exception as e:
        pytest.fail(f"doSearch() raised an exception: {str(e)}")
