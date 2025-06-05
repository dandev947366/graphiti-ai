from sentence_transformers import SentenceTransformer


class MxbaiEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("mxbai/mxbai-embed-large")

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings
