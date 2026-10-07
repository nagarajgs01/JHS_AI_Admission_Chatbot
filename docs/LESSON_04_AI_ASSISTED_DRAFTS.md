# Lesson 04: AI-Assisted Admin Drafts

The suggestion endpoint is a staff productivity feature, not an autonomous answer
channel. It never sends messages and it never publishes knowledge.

## Grounded path

For a question supported by published knowledge, retrieval supplies approved passages
to the local model. The model creates up to three alternative drafts. Every draft is
then checked by the grounding verifier; unsupported drafts are discarded.

## No-evidence path

When retrieval has no directly supporting evidence, the model is not asked to invent
an answer. The API returns only deterministic callback drafts that tell the parent the
school must confirm the requested information.

## Admin contract

`POST /v1/admin/unanswered/{id}/suggestions?school_id=...` requires the admin key.
Every returned item has a `kind`, a `requires_staff_verification` flag, and source
citations when it is grounded. Staff must still edit or approve a draft through the
separate response endpoints.
