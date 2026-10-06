# Lesson 2: Persisting embeddings in PostgreSQL and pgvector

## Goal

Move reviewed JHS knowledge from JSON and application memory into durable,
tenant-scoped PostgreSQL tables.

## Ingestion transaction

The ingestion command performs one database transaction:

1. upsert the school by slug;
2. derive stable UUIDs for reviewed knowledge entries;
3. upsert each entry and its approval status;
4. split each entry into chunks;
5. create local 384-dimensional MiniLM embeddings;
6. replace the entry's previous chunks;
7. commit only after all records succeed.

Stable entry IDs make the command idempotent: rerunning it updates existing JHS
records rather than creating duplicates.

## Run from the project root

```bash
set -a
source .env
set +a
PYTHONPATH=apps/api python -m app.scripts.ingest_seed
```

## Search pgvector

```bash
PYTHONPATH=apps/api python -m app.scripts.search_knowledge \
  "Does the school offer aquatic activities?"
```

Retrieval is filtered by school slug and `published` status before ranking. The
score combines cosine similarity from pgvector with PostgreSQL full-text rank.

## Inspect stored records

```bash
docker compose exec db psql -U admissions -d admissions -c \
  "SELECT count(*) FROM knowledge_entries;"

docker compose exec db psql -U admissions -d admissions -c \
  "SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding)) FROM knowledge_chunks;"
```
