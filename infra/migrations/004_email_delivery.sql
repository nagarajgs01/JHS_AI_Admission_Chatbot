ALTER TABLE question_responses
ADD COLUMN IF NOT EXISTS delivery_status text NOT NULL DEFAULT 'not_applicable',
ADD COLUMN IF NOT EXISTS delivery_error text,
ADD COLUMN IF NOT EXISTS delivered_at timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'question_responses_delivery_status_check'
    ) THEN
        ALTER TABLE question_responses
        ADD CONSTRAINT question_responses_delivery_status_check
        CHECK (delivery_status IN ('pending', 'sent', 'failed', 'not_applicable'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS question_responses_delivery_status_idx
ON question_responses (delivery_status)
WHERE delivery_status IN ('pending', 'failed');
