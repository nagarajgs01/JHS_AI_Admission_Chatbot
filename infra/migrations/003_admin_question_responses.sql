CREATE TABLE IF NOT EXISTS question_responses (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    unanswered_question_id uuid NOT NULL
        REFERENCES unanswered_questions(id) ON DELETE CASCADE,
    answer text NOT NULL CHECK (length(btrim(answer)) >= 2),
    status text NOT NULL CHECK (status IN ('draft', 'approved')),
    created_at timestamptz NOT NULL DEFAULT now(),
    approved_at timestamptz
);

CREATE INDEX IF NOT EXISTS question_responses_question_created_idx
ON question_responses (unanswered_question_id, created_at DESC);

ALTER TABLE question_responses ENABLE ROW LEVEL SECURITY;
