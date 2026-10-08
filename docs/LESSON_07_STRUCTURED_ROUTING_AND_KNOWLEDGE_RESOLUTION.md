# Lesson 07: Structured Routing and Knowledge Resolution

Qwen now performs a non-answering understanding pass before retrieval. It returns a
validated structure containing the normalized question, school/out-of-scope decision,
clarity, intent, and requested facts. Retrieval operates on the normalized wording,
while the original wording remains the user-facing question. Deterministic rules remain
only as a failure-safe when structured model output is unavailable.

Admin answer generation evaluates each retrieved source independently. Two conflicting
sources therefore remain two separate grounded drafts with separate citations instead
of being blended into one answer. Identical drafts may combine citations.

On approval, staff choose between resolving only the individual enquiry and publishing
the verified response as reusable knowledge. Published resolutions receive optional
validity dates, are embedded through the existing ingestion pipeline, and are linked to
the unanswered question. Staff may explicitly select outdated source entries to archive;
the system never archives candidates automatically.

The shared development API key remains a baseline mechanism. Production still requires
individual staff identities, role-based approval, a fully atomic publication workflow,
and audit events for every archive and publication action.
