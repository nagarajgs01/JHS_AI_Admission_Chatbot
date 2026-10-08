import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import KnowledgeEntry, KnowledgeStatus


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "at", "available", "be", "can", "do", "does", "for",
    "from", "have", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "school", "tell", "the", "there", "to", "what", "when", "where",
    "which", "who", "with", "you", "through", "this", "jhs", "our",
    "while", "will", "during", "happen", "happens", "get", "time",
    "use", "uses", "used", "idea", "ideas",
}


def normalize_token(token: str) -> str:
    aliases = {
        "fees": "fee",
        "fess": "fee",
        "feee": "fee",
        "grades": "grade",
        "classes": "class",
        "learning": "learn",
        "learns": "learn",
        "learned": "learn",
        "playing": "play",
        "played": "play",
        "plays": "play",
        "children": "child",
        "syllabus": "curriculum",
        "syllabuses": "curriculum",
        "ages": "age",
        "assessed": "assessment",
        "assessing": "assessment",
        "assessments": "assessment",
        "options": "option",
        "transportation": "transport",
        "transports": "transport",
        "buses": "bus",
        "studies": "study",
    }
    return aliases.get(token, token)


def tokenize(value: str) -> set[str]:
    return {
        normalize_token(token)
        for token in TOKEN_RE.findall(value.lower())
        if token not in STOP_WORDS
    }


@dataclass(frozen=True)
class SearchHit:
    entry: KnowledgeEntry
    score: float
    evidence_supported: bool = False


class InMemoryKnowledgeRepository:
    """Development repository; production will use PostgreSQL + pgvector."""

    def __init__(self, entries: list[KnowledgeEntry] | None = None) -> None:
        self.entries = entries or []

    def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        self.entries.append(entry)
        return entry

    @staticmethod
    def _is_active(entry: KnowledgeEntry) -> bool:
        now = datetime.now(timezone.utc)
        return (
            (entry.valid_from is None or entry.valid_from <= now)
            and (entry.expires_at is None or entry.expires_at > now)
        )

    def search(self, school_id: str, query: str, limit: int = 4) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        hits: list[SearchHit] = []
        for entry in self.entries:
            if (
                entry.school_id != school_id
                or entry.status != KnowledgeStatus.PUBLISHED
                or not self._is_active(entry)
            ):
                continue
            document_tokens = tokenize(f"{entry.title} {entry.content} {' '.join(entry.tags)}")
            overlap = len(query_tokens & document_tokens)
            if overlap == 0:
                continue
            # Normalized lexical score used only for the runnable first slice.
            query_coverage = overlap / len(query_tokens)
            title_overlap = len(query_tokens & tokenize(entry.title))
            title_coverage = title_overlap / len(query_tokens)
            score = min(1.0, (0.75 * query_coverage) + (0.25 * title_coverage))
            hits.append(SearchHit(entry=entry, score=round(score, 4)))

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def clarification_candidates(
        self,
        school_id: str,
        query: str,
        limit: int = 3,
    ) -> list[SearchHit]:
        from difflib import SequenceMatcher

        query_tokens = tokenize(query)
        candidates: list[SearchHit] = []
        for entry in self.entries:
            if (
                entry.school_id != school_id
                or entry.status != KnowledgeStatus.PUBLISHED
                or not self._is_active(entry)
            ):
                continue
            document_tokens = tokenize(f"{entry.title} {' '.join(entry.tags)}")
            exact = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            fuzzy = max(
                (
                    SequenceMatcher(None, left, right).ratio()
                    for left in query_tokens
                    for right in document_tokens
                ),
                default=0.0,
            )
            score = max(exact, fuzzy * 0.55)
            if score >= 0.35:
                candidates.append(SearchHit(entry=entry, score=round(score, 4)))
        return sorted(candidates, key=lambda hit: hit.score, reverse=True)[:limit]


class InMemoryHybridKnowledgeRepository(InMemoryKnowledgeRepository):
    """Hybrid chunk retrieval used before the PostgreSQL adapter is enabled."""

    def __init__(self, entries, embedding_model) -> None:
        super().__init__(entries)
        from .ingestion import embed_entries

        self.embedding_model = embedding_model
        self.chunks = embed_entries(self._published_entries(), embedding_model)

    def _published_entries(self) -> list[KnowledgeEntry]:
        return [
            entry
            for entry in self.entries
            if entry.status == KnowledgeStatus.PUBLISHED and self._is_active(entry)
        ]

    def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        from .ingestion import embed_entries

        result = super().add(entry)
        if entry.status == KnowledgeStatus.PUBLISHED:
            self.chunks.extend(embed_entries([entry], self.embedding_model))
        return result

    def search(self, school_id: str, query: str, limit: int = 4) -> list[SearchHit]:
        from .embeddings import cosine_similarity

        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_vector = self.embedding_model.embed([query])[0]
        entries_by_id = {entry.id: entry for entry in self._published_entries()}
        scores: dict = {}

        for chunk in self.chunks:
            if chunk.school_id != school_id:
                continue
            chunk_tokens = tokenize(
                f"{chunk.title} {chunk.content} {' '.join(chunk.tags)}"
            )
            lexical = len(query_tokens & chunk_tokens) / len(query_tokens)
            semantic = max(0.0, cosine_similarity(query_vector, chunk.embedding))
            if lexical == 0 and semantic < 0.72:
                continue
            score = min(1.0, (0.65 * lexical) + (0.35 * semantic))
            scores[chunk.knowledge_id] = max(scores.get(chunk.knowledge_id, 0.0), score)

        hits = [
            SearchHit(entry=entries_by_id[entry_id], score=round(score, 4))
            for entry_id, score in scores.items()
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
