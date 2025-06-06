"""
This script tests the functionality of a HuggingFace-based embedding utility by
generating embeddings for single and multiple input texts using the `HuggingFaceEmbedder`
class and printing their shapes to verify correctness.

Functionality:
- Instantiates a HuggingFaceEmbedder.
- Embeds a single text string and prints the resulting embedding shape.
- Embeds multiple text strings and prints the number of returned embeddings
  and shape of the first embedding.

Requirements:
- `llm_embedder.py` with a `HuggingFaceEmbedder` class implementing an async `create` method.
- Transformers and tokenizers dependencies for HuggingFace models.
- Python 3.7+ (for `asyncio.run` support).

Use Case:
Primarily used for validating embedding model outputs in NLP pipelines.
Useful in systems for semantic search, clustering, RAG, or AI-powered data indexing
before inserting into vector databases or downstream applications.
"""

import asyncio
import os
from llm_embedder import HuggingFaceEmbedder


async def test_embedding_and_insert():
    embedder = HuggingFaceEmbedder()

    # Test single string input
    single_text = "Hello world, this is a test."
    print("Testing single input embedding and DB insert...")
    embeddings = await embedder.create(single_text)
    print(f"Embedding shape: {embeddings[0].shape}")

    # Test multiple strings input
    multiple_texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
    ]
    print("Testing multiple inputs embedding and DB insert...")
    embeddings = await embedder.create(multiple_texts)
    print(f"Number of embeddings returned: {len(embeddings)}")
    print(f"Shape of first embedding: {embeddings[0].shape}")


if __name__ == "__main__":
    asyncio.run(test_embedding_and_insert())
