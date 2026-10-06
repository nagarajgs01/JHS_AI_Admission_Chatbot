CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE schools (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug text UNIQUE NOT NULL,
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TYPE knowledge_status AS ENUM ('draft', 'published', 'archived');

CREATE TABLE knowledge_entries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    school_id uuid NOT NULL REFERENCES schools(id),
    title text NOT NULL,
    content text NOT NULL,
    source_label text NOT NULL,
    source_url text,
    status knowledge_status NOT NULL DEFAULT 'draft',
    tags text[] NOT NULL DEFAULT '{}',
    version integer NOT NULL DEFAULT 1,
    approved_by uuid,
    approved_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE knowledge_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    school_id uuid NOT NULL REFERENCES schools(id),
    knowledge_entry_id uuid NOT NULL REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    token_count integer NOT NULL,
    embedding vector(384),
    search_document tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (knowledge_entry_id, chunk_index)
);

CREATE INDEX knowledge_chunks_embedding_idx ON knowledge_chunks
USING hnsw (embedding vector_cosine_ops);
CREATE INDEX knowledge_chunks_text_idx ON knowledge_chunks USING gin (search_document);
CREATE INDEX knowledge_entries_school_status_idx ON knowledge_entries (school_id, status);

CREATE TABLE unanswered_questions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    school_id uuid NOT NULL REFERENCES schools(id),
    question text NOT NULL,
    contact_email text,
    consent_to_contact boolean NOT NULL DEFAULT false,
    status text NOT NULL DEFAULT 'open',
    resolution_knowledge_entry_id uuid REFERENCES knowledge_entries(id),
    created_at timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);

CREATE TABLE admission_leads (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    school_id uuid NOT NULL REFERENCES schools(id),
    student_name text NOT NULL,
    grade_applied_for text NOT NULL,
    guardian_name text NOT NULL,
    mobile_number text NOT NULL,
    email text NOT NULL,
    preferred_contact text,
    consent_to_contact boolean NOT NULL CHECK (consent_to_contact),
    status text NOT NULL DEFAULT 'new',
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE knowledge_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE unanswered_questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE admission_leads ENABLE ROW LEVEL SECURITY;
