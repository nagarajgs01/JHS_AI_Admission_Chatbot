from dataclasses import dataclass
from uuid import UUID, NAMESPACE_URL, uuid5

from pgvector import Vector

from .database import database_connection
from .embeddings import EmbeddingModel
from .ingestion import embed_entries
from .models import KnowledgeEntry
from .retrieval import SearchHit
from .retrieval_policy import (
    evaluate_evidence,
    expand_query,
    meaningful_keyword_score,
    synonym_evidence_score,
)


@dataclass(frozen=True)
class IngestionResult:
    school_id: UUID
    entries_written: int
    chunks_written: int


class PostgresKnowledgeRepository:
    """Chat-facing adapter around the tested PostgreSQL retrieval pipeline."""

    def __init__(self, embedding_model: EmbeddingModel, minimum_score: float = 0.38) -> None:
        self.embedding_model = embedding_model
        self.minimum_score = minimum_score

    def search(self, school_id: str, query: str, limit: int = 4) -> list[SearchHit]:
        expanded_query = expand_query(query)
        query_embedding = self.embedding_model.embed([expanded_query])[0]
        rows = search_entries(school_id, query, query_embedding, limit)
        if not rows:
            return []

        supported_rows = []
        for row in rows:
            passage = f"{row['title']} {row['content']} {' '.join(row['tags'])}"
            decision = evaluate_evidence(
                query,
                passage,
                float(row["combined_score"]),
                self.minimum_score,
            )
            if decision.supported:
                supported_rows.append(row)

        if not supported_rows:
            return []

        return [
            SearchHit(
                entry=KnowledgeEntry(
                    id=row["id"],
                    school_id=school_id,
                    title=row["title"],
                    content=row["content"],
                    source_label=row["source_label"],
                    source_url=row["source_url"],
                    status="published",
                    tags=row["tags"],
                ),
                score=round(float(row["combined_score"]), 4),
                evidence_supported=(index == 0),
            )
            for index, row in enumerate(supported_rows)
        ]


def stable_entry_id(entry: KnowledgeEntry) -> UUID:
    identity = f"{entry.school_id}|{entry.title}|{entry.source_label}"
    return uuid5(NAMESPACE_URL, identity)


def ingest_entries(
    school_slug: str,
    school_name: str,
    entries: list[KnowledgeEntry],
    embedding_model: EmbeddingModel,
) -> IngestionResult:
    """Upsert reviewed entries and their vectors in one transaction."""

    if any(entry.school_id != school_slug for entry in entries):
        raise ValueError("Every entry must belong to the requested school")

    entries_written = 0
    chunks_written = 0

    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO schools (slug, name)
                VALUES (%s, %s)
                ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                (school_slug, school_name),
            )
            school_id = cursor.fetchone()[0]

            for original_entry in entries:
                entry_id = stable_entry_id(original_entry)
                entry = original_entry.model_copy(update={"id": entry_id})
                cursor.execute(
                    """
                    INSERT INTO knowledge_entries (
                        id, school_id, title, content, source_label,
                        source_url, status, tags, approved_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                            CASE WHEN %s = 'published' THEN now() ELSE NULL END, now())
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        source_label = EXCLUDED.source_label,
                        source_url = EXCLUDED.source_url,
                        status = EXCLUDED.status,
                        tags = EXCLUDED.tags,
                        approved_at = EXCLUDED.approved_at,
                        updated_at = now()
                    """,
                    (
                        entry.id,
                        school_id,
                        entry.title,
                        entry.content,
                        entry.source_label,
                        entry.source_url,
                        entry.status.value,
                        entry.tags,
                        entry.status.value,
                    ),
                )
                entries_written += 1

                chunks = embed_entries([entry], embedding_model)
                cursor.execute(
                    "DELETE FROM knowledge_chunks WHERE knowledge_entry_id = %s",
                    (entry.id,),
                )
                for chunk in chunks:
                    cursor.execute(
                        """
                        INSERT INTO knowledge_chunks (
                            school_id, knowledge_entry_id, chunk_index,
                            content, token_count, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            school_id,
                            entry.id,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.token_count,
                            chunk.embedding,
                        ),
                    )
                    chunks_written += 1

        connection.commit()

    return IngestionResult(
        school_id=school_id,
        entries_written=entries_written,
        chunks_written=chunks_written,
    )


def search_entries(
    school_slug: str,
    query: str,
    query_embedding: list[float],
    limit: int = 5,
) -> list[dict]:
    """Run tenant-scoped, published-only hybrid retrieval in PostgreSQL."""

    query_vector = Vector(query_embedding)

    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    ke.id,
                    ke.title,
                    kc.content,
                    ke.source_label,
                    ke.source_url,
                    ke.tags,
                    1 - (kc.embedding <=> %s) AS semantic_score
                FROM knowledge_chunks kc
                JOIN knowledge_entries ke ON ke.id = kc.knowledge_entry_id
                JOIN schools s ON s.id = kc.school_id
                WHERE s.slug = %s
                  AND ke.status = 'published'
                  AND kc.embedding IS NOT NULL
                ORDER BY semantic_score DESC
                LIMIT %s
                """,
                (query_vector, school_slug, max(limit, 20)),
            )
            columns = [description.name for description in cursor.description]
            results = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    for result in results:
        searchable_text = (
            f"{result['title']} {result['content']} {' '.join(result['tags'])}"
        )
        keyword_score = meaningful_keyword_score(query, searchable_text)
        synonym_score = synonym_evidence_score(query, searchable_text)
        result["keyword_score"] = keyword_score
        result["synonym_score"] = synonym_score
        result["combined_score"] = (
            0.65 * float(result["semantic_score"])
            + 0.25 * keyword_score
            + 0.10 * synonym_score
        )

    return sorted(results, key=lambda item: item["combined_score"], reverse=True)[:limit]
