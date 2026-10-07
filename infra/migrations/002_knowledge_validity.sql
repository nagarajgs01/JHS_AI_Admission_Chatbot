ALTER TABLE knowledge_entries
ADD COLUMN IF NOT EXISTS valid_from timestamptz,
ADD COLUMN IF NOT EXISTS expires_at timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'knowledge_entries_validity_window'
    ) THEN
        ALTER TABLE knowledge_entries
        ADD CONSTRAINT knowledge_entries_validity_window
        CHECK (
            valid_from IS NULL
            OR expires_at IS NULL
            OR expires_at > valid_from
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS knowledge_entries_validity_idx
ON knowledge_entries (school_id, status, valid_from, expires_at);
