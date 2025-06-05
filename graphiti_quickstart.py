import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Configure logging
logging.basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")


async def main():
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Build indices and constraints once
        await graphiti.build_indices_and_constraints()
        logger.info("Built indices and constraints")

        episodes = [
            {
                "content": "Kamala Harris is the Attorney General of California. She was previously "
                "the district attorney for San Francisco.",
                "type": EpisodeType.text,
                "description": "podcast transcript",
            },
            {
                "content": "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
                "type": EpisodeType.text,
                "description": "podcast transcript",
            },
            {
                "content": {
                    "name": "Gavin Newsom",
                    "position": "Governor",
                    "state": "California",
                    "previous_role": "Lieutenant Governor",
                    "previous_location": "San Francisco",
                },
                "type": EpisodeType.json,
                "description": "podcast metadata",
            },
            {
                "content": {
                    "name": "Gavin Newsom",
                    "position": "Governor",
                    "term_start": "January 7, 2019",
                    "term_end": "Present",
                },
                "type": EpisodeType.json,
                "description": "podcast metadata",
            },
        ]

        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f"Freakonomics Radio {i}",
                episode_body=(
                    episode["content"]
                    if isinstance(episode["content"], str)
                    else json.dumps(episode["content"])
                ),
                source=episode["type"],
                source_description=episode["description"],
                reference_time=datetime.now(timezone.utc),
            )
            logger.info(
                f'Added episode: Freakonomics Radio {i} ({episode["type"].value})'
            )

    finally:
        await graphiti.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
