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
