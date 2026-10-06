import re
from dataclasses import dataclass
from uuid import UUID

from .embeddings import EmbeddingModel
from .models import KnowledgeEntry
from .retrieval import tokenize


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EmbeddedChunk:
    knowledge_id: UUID
    school_id: str
    title: str
    source_label: str
    source_url: str | None
    content: str
    chunk_index: int
    token_count: int
    tags: tuple[str, ...]
    embedding: list[float]


def chunk_text(text: str, max_words: int = 120, overlap_sentences: int = 1) -> list[str]:
    """Split on sentence boundaries while retaining small contextual overlap."""

    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(item.split()) for item in current)
        current.append(sentence)
        current_words += sentence_words
    if current:
        candidate = " ".join(current)
        if not chunks or candidate != chunks[-1]:
            chunks.append(candidate)
    return chunks


def embed_entries(entries: list[KnowledgeEntry], model: EmbeddingModel) -> list[EmbeddedChunk]:
    pending: list[tuple[KnowledgeEntry, int, str]] = []
    for entry in entries:
        for index, content in enumerate(chunk_text(entry.content)):
            pending.append((entry, index, content))

    vectors = model.embed(
        [f"{entry.title}. {content} Tags: {', '.join(entry.tags)}" for entry, _, content in pending]
    )
    return [
        EmbeddedChunk(
            knowledge_id=entry.id,
            school_id=entry.school_id,
            title=entry.title,
            source_label=entry.source_label,
            source_url=entry.source_url,
            content=content,
            chunk_index=index,
            token_count=len(tokenize(content)),
            tags=tuple(entry.tags),
            embedding=vector,
        )
        for (entry, index, content), vector in zip(pending, vectors, strict=True)
    ]

