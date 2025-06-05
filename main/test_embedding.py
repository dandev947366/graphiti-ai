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
