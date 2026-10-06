import hashlib
import math
from typing import Protocol

from .retrieval import tokenize


class EmbeddingModel(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbeddingModel:
    """Small deterministic embedder for development and unit tests.

    It keeps local development offline. Production uses
    SentenceTransformerEmbeddingModel for true semantic similarity.
    """

    dimensions = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class SentenceTransformerEmbeddingModel:
    """Local production embedder; question and school text stay in our runtime."""

    dimensions = 384

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))

