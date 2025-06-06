from typing import Iterable, List
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import singlestoredb as s2
import numpy as np

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "dunzhang/stella_en_1.5B_v5"


def floats_to_blob(floats):
    arr = np.array(floats, dtype=np.float32)
    return arr.tobytes()


def read_local_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, max_length: int = 512) -> list[str]:
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


class HuggingFaceEmbedderConfig(EmbedderConfig):
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str | None = None


class HuggingFaceEmbedder(EmbedderClient):
    """
    HuggingFace Embedder Client
    """

    def __init__(self, config: HuggingFaceEmbedderConfig | None = None):
        if config is None:
            config = HuggingFaceEmbedderConfig()
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        self.conn = s2.connect(os.environ.get("SINGLESTORE_URL"))

    @staticmethod
    def blob_to_floats(blob):
        return np.frombuffer(blob, dtype=np.float32)

    async def create(self, input):
        embeddings = self.model.encode(input)
        with self.conn.cursor() as cur:
            if isinstance(input, str):
                texts = [input]
                embeddings = [embeddings]
            else:
                texts = input

            for text, emb in zip(texts, embeddings):
                vector_blob = floats_to_blob(emb)
                sql = "INSERT INTO myvectortable (text, vector) VALUES (%s, %s)"
                cur.execute(sql, (text, vector_blob))
        self.conn.commit()  # Make sure to commit
        return embeddings


def main(filepath: str):
    embedder = HuggingFaceEmbedder()
    text = read_local_file(filepath)
    chunks = chunk_text(text)
    import asyncio

    embeddings = asyncio.run(embedder.create(chunks))

    print(f"Embedded and stored {len(chunks)} chunks from file: {filepath}")


if __name__ == "__main__":
    local_file_path = "./test_embedding.txt"
    main(local_file_path)
