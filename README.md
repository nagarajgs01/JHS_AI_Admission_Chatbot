# AI Admissions Assistant

A secure, multi-school admissions assistant that answers only from approved knowledge, captures admission enquiries, and escalates questions when evidence is insufficient.

## Architecture

- `apps/api`: FastAPI API and rules-first RAG orchestration
- `apps/web`: React + TypeScript application containing the embeddable chat experience and admin foundation
- `infra`: PostgreSQL/pgvector local infrastructure
- `docs`: architecture, API behavior, and development roadmap

The first vertical slice is intentionally safe: retrieve approved content, apply an evidence threshold, generate a grounded answer through a replaceable LLM adapter, or return a fixed escalation response. The API never performs open-web search.

## Quick start

1. Install Ollama, then run `ollama pull qwen3:4b`.
2. Copy `.env.example` to `.env`.
3. Start PostgreSQL: `docker compose up -d db`.
4. Create a Python environment and install `apps/api/requirements-dev.txt`.
5. Run the API from `apps/api`: `uvicorn app.main:app --reload`.
6. Install and run the web app from `apps/web`: `npm install && npm run dev`.

The API retrieves reviewed JHS brochure knowledge from PostgreSQL/pgvector by default.
`LLM_PROVIDER=ollama` uses the local Qwen model; `LLM_PROVIDER=mock` runs deterministic
tests without a model server. Set `KNOWLEDGE_PROVIDER=memory` only for isolated local tests.
Production persistence is represented by the SQL migration in `infra/migrations`.

Read `docs/LESSON_01_LOCAL_LLM.md` before changing the model adapter.

## Ingest reviewed JHS knowledge

After PostgreSQL is healthy and the local embedding dependencies are installed:

```bash
set -a
source .env
set +a
PYTHONPATH=apps/api python -m app.scripts.ingest_seed
```

Test pgvector retrieval:

```bash
PYTHONPATH=apps/api python -m app.scripts.search_knowledge \
  "Does the school offer aquatic activities?"
```

See `docs/LESSON_02_POSTGRES_INGESTION.md` for the design and verification steps.

## Safety guarantees in the initial implementation

- Only `published` knowledge entries are searchable.
- Every request is scoped by `school_id`.
- Low-evidence questions receive a deterministic escalation response.
- The model adapter receives only the question and retrieved approved passages.
- No web-search tool is available to the model.
- The configured model is self-hosted; no OpenAI API is used.
- Email collection is a separate, explicit-consent operation.
