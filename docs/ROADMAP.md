# Delivery roadmap

## Milestone 1 — Secure RAG foundation (current)

- Monorepo and local infrastructure
- Tenant-scoped, published-only retrieval contract
- Evidence threshold and deterministic escalation
- Self-hosted model adapters for Ollama development and vLLM production
- Basic React chat experience
- Initial database schema

Completed for the JHS baseline: the official brochure has been converted into reviewed,
source-labelled knowledge records, and hybrid retrieval now uses chunk-level local embeddings.

## Milestone 2 — Production knowledge base

- Admin authentication and role-based access
- PostgreSQL repositories and tenant row-level-security policies
- Document upload, text extraction, cleaning, chunking, local embeddings, and versioning
- Hybrid retrieval: vector similarity + PostgreSQL full-text search
- Reranking and evaluation dataset
- Draft → review → publish workflow
- Resolution flow that converts staff-approved answers into searchable knowledge

## Milestone 3 — Admissions workflows

- Configurable admission enquiry form
- Full application workflow with save/resume
- Secure object storage and malware scanning
- OCR extraction with parent confirmation
- Email notifications and CSV export

## Milestone 4 — Operations and deployment

- Audit logs, metrics, tracing, rate limits, abuse controls, and PII redaction
- Automated RAG quality, tenant-isolation, security, accessibility, and load tests
- CI/CD and staged cloud deployment
- Embeddable Web Component SDK with allowed-domain controls

## Decisions/materials needed from the school

- Exact official school website URL and approved public pages
- Admission handbook, fee structure, grade/age eligibility, dates, timings, transport, facilities, policies, and FAQs
- Enquiry/application fields and validation rules
- Required documents, retention duration, and authorized staff roles
- Escalation mailbox, service hours, consent wording, branding, languages, and integration/API details
