"""
This script tests the `HuggingFaceEmbedder` class with both a single string and a list of strings.
It uses the `HuggingFaceEmbedderConfig` for configuration and prints the shape of the resulting
embeddings for verification.

Functionality:
- Instantiates `HuggingFaceEmbedder` using a configuration object.
- Embeds a single string and prints the shape of the output.
- Embeds a list of strings and prints the shape of the output.

Requirements:
- `llm_embedder.py` file containing:
  - `HuggingFaceEmbedder` class with async `create` method.
  - `HuggingFaceEmbedderConfig` class for configuration.
- HuggingFace transformers and tokenizers.
- Python 3.7+ with asyncio support.

Use Case:
Useful for verifying that text embeddings are correctly generated with expected
output shapes. Typically used in applications like semantic search, recommendation
systems, RAG pipelines, or any system requiring vector representations of text.
"""

import asyncio
from typing import List
from llm_embedder import HuggingFaceEmbedder, HuggingFaceEmbedderConfig


async def test_embedder_with_string():
    config = HuggingFaceEmbedderConfig()
    embedder = HuggingFaceEmbedder(config)

    input_text = "This is a test sentence."
    shape = await embedder.create(input_text)
    print(f"Embedding shape: {shape}")


async def test_embedder_with_list():
    config = HuggingFaceEmbedderConfig()
    embedder = HuggingFaceEmbedder(config)

    input_texts: List[str] = ["This is the first sentence.", "Here's another one."]
    shape = await embedder.create(input_texts)
    print(f"Embedding shape: {shape}")


if __name__ == "__main__":
    asyncio.run(test_embedder_with_string())
    asyncio.run(test_embedder_with_list())
