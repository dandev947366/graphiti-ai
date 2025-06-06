from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from datetime import datetime
from main.llm_embedder import HuggingFaceEmbedder
from main.local_ai_client import LocalAiClient
import asyncio
import os


async def doSearch():
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

    episodes = [
        "Kamala Harris is the Attorney General of California. She was previously "
        "the district attorney for San Francisco.",
        "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
    ]
    print("embbed episodes")
    for i, episode in enumerate(episodes):
        await graphiti.add_episode(
            name=f"Freakonomics Radio {i}",
            episode_body=episode,
            source=EpisodeType.text,
            source_description="podcast",
            reference_time=datetime.now(),
        )

    results = await graphiti.search("Who was the California Attorney General?")
    await graphiti.close()

    print("Results: ")
    print(results)


asyncio.run(doSearch())
