# Lesson 03: School Admin Question Queue

This milestone turns stored fallback questions into a reviewable school workflow.

## Security boundary

Parent chat endpoints remain public. Admin endpoints require the `X-Admin-Key`
header. The comparison uses a constant-time function, and non-development
environments refuse to use the built-in local placeholder key.

## Data model

`unanswered_questions` stores the original parent question and contact consent.
`question_responses` stores an append-only history of draft and approved answers.
Saving a new draft does not overwrite an older draft. Approving an answer inserts
an approved response and marks the question as `answered` in one transaction.

## Endpoints

- `GET /v1/admin/unanswered` lists a school's questions by status.
- `GET /v1/admin/unanswered/{id}` returns one question and its latest response.
- `POST /v1/admin/unanswered/{id}/draft` saves a draft response.
- `POST /v1/admin/unanswered/{id}/approve` stores the approved response.

Email delivery and knowledge-base publication are intentionally separate future
operations. This prevents an email or embedding failure from losing the approved
answer.
