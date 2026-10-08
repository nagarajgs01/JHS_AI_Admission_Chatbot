from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .embeddings import HashingEmbeddingModel, SentenceTransformerEmbeddingModel
from .email_service import build_email_sender
from .llm import EvidenceOnlyModel, OllamaModel, VLLMModel
from .models import (
    AdminAnswerSubmission,
    AdminQuestionResponse,
    AdminSuggestionResponse,
    AdminUnansweredQuestion,
    ChatRequest,
    ChatResponse,
    EscalationContact,
    KnowledgeEntry,
    Lead,
    LeadSubmission,
)
from .postgres_knowledge import PostgresKnowledgeRepository
from .postgres_operations import PostgresLeadRepository, PostgresUnansweredRepository
from .retrieval import InMemoryHybridKnowledgeRepository
from .seed import load_published_seed_data
from .service import (
    AdminSuggestionService,
    ChatService,
    LeadRepository,
    UnansweredRepository,
    answer_has_placeholders,
)


settings = get_settings()
embedding_model = (
    SentenceTransformerEmbeddingModel(settings.embedding_model)
    if settings.embedding_provider == "sentence-transformers"
    else HashingEmbeddingModel()
)
knowledge = (
    PostgresKnowledgeRepository(embedding_model, settings.answer_min_score)
    if settings.knowledge_provider == "postgres"
    else InMemoryHybridKnowledgeRepository(load_published_seed_data(), embedding_model)
)
if settings.knowledge_provider == "postgres":
    unanswered = PostgresUnansweredRepository()
    leads = PostgresLeadRepository()
else:
    unanswered = UnansweredRepository()
    leads = LeadRepository()


def build_model():
    if settings.llm_provider == "ollama":
        return OllamaModel(settings.llm_base_url, settings.llm_model, settings.llm_timeout_seconds)
    if settings.llm_provider == "vllm":
        return VLLMModel(settings.llm_base_url, settings.llm_model, settings.llm_timeout_seconds)
    return EvidenceOnlyModel()


chat_service = ChatService(knowledge, unanswered, build_model(), settings.answer_min_score)
language_model = chat_service.model
admin_suggestion_service = AdminSuggestionService(
    knowledge,
    language_model,
    settings.answer_min_score,
)
email_sender = build_email_sender(settings)


async def deliver_approved_answer(item, response: AdminQuestionResponse) -> AdminQuestionResponse:
    if not item.contact_email or not item.consent_to_contact:
        return response
    if not email_sender.enabled:
        error = "Email delivery is not configured"
        unanswered.update_delivery(response.id, "failed", error)
        return response.model_copy(update={"delivery_status": "failed", "delivery_error": error})
    try:
        await email_sender.send_parent_answer(
            item.contact_email,
            item.question,
            response.answer,
        )
        unanswered.update_delivery(response.id, "sent")
        refreshed = unanswered.get_for_school(item.school_id, item.id)
        return response.model_copy(
            update={
                "delivery_status": "sent",
                "delivered_at": refreshed.delivered_at if refreshed else None,
            }
        )
    except Exception as error:  # SMTP/network errors must not roll back approval.
        message = str(error)[:500] or error.__class__.__name__
        unanswered.update_delivery(response.id, "failed", message)
        return response.model_copy(update={"delivery_status": "failed", "delivery_error": message})


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="AI Admissions Assistant API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Admin-Key"],
)


def require_admin_key(
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
) -> None:
    if (
        settings.app_env != "development"
        and settings.admin_api_key == "change-me-local-only"
    ):
        raise HTTPException(status_code=503, detail="Admin authentication is not configured")
    if x_admin_key is None or not compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin credentials")


AdminAccess = Annotated[None, Depends(require_admin_key)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/model")
async def model_health() -> dict[str, str]:
    healthy = await language_model.health()
    if not healthy:
        raise HTTPException(status_code=503, detail="Local language model is unavailable")
    return {"status": "ok", "provider": settings.llm_provider, "model": settings.llm_model}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await chat_service.respond(request)


@app.post("/v1/escalations/contact", status_code=status.HTTP_204_NO_CONTENT)
async def save_escalation_contact(contact: EscalationContact) -> None:
    item = unanswered.get(contact.unanswered_id)
    if item is None or item.school_id != contact.school_id:
        raise HTTPException(status_code=404, detail="Unanswered question not found")
    if not contact.consent_to_contact:
        raise HTTPException(status_code=422, detail="Consent is required")
    unanswered.attach_contact(contact.unanswered_id, contact.email)


@app.post("/v1/leads", response_model=Lead, status_code=status.HTTP_201_CREATED)
async def create_lead(submission: LeadSubmission) -> Lead:
    return leads.create(submission)


@app.get(
    "/v1/admin/unanswered",
    response_model=list[AdminUnansweredQuestion],
)
async def list_unanswered_questions(
    _: AdminAccess,
    school_id: str,
    question_status: Annotated[
        Literal["open", "answered", "closed"], Query(alias="status")
    ] = "open",
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AdminUnansweredQuestion]:
    if not isinstance(unanswered, PostgresUnansweredRepository):
        raise HTTPException(status_code=501, detail="Admin queue requires PostgreSQL")
    return unanswered.list_for_school(school_id, question_status, limit, offset)


@app.get(
    "/v1/admin/unanswered/{unanswered_id}",
    response_model=AdminUnansweredQuestion,
)
async def get_unanswered_question(
    unanswered_id: UUID,
    school_id: str,
    _: AdminAccess,
) -> AdminUnansweredQuestion:
    if not isinstance(unanswered, PostgresUnansweredRepository):
        raise HTTPException(status_code=501, detail="Admin queue requires PostgreSQL")
    item = unanswered.get_for_school(school_id, unanswered_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Unanswered question not found")
    return item


@app.post(
    "/v1/admin/unanswered/{unanswered_id}/suggestions",
    response_model=AdminSuggestionResponse,
)
async def generate_answer_suggestions(
    unanswered_id: UUID,
    school_id: str,
    _: AdminAccess,
) -> AdminSuggestionResponse:
    if not isinstance(unanswered, PostgresUnansweredRepository):
        raise HTTPException(status_code=501, detail="Admin queue requires PostgreSQL")
    item = unanswered.get_for_school(school_id, unanswered_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Unanswered question not found")
    if item.status != "open":
        raise HTTPException(status_code=409, detail="Question is no longer open")
    return await admin_suggestion_service.generate(
        school_id,
        unanswered_id,
        item.question,
    )


@app.post(
    "/v1/admin/unanswered/{unanswered_id}/draft",
    response_model=AdminQuestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def save_answer_draft(
    unanswered_id: UUID,
    submission: AdminAnswerSubmission,
    _: AdminAccess,
) -> AdminQuestionResponse:
    if not isinstance(unanswered, PostgresUnansweredRepository):
        raise HTTPException(status_code=501, detail="Admin queue requires PostgreSQL")
    try:
        return unanswered.save_answer(
            submission.school_id,
            unanswered_id,
            submission.answer,
            "draft",
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/v1/admin/unanswered/{unanswered_id}/approve",
    response_model=AdminQuestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def approve_answer(
    unanswered_id: UUID,
    submission: AdminAnswerSubmission,
    _: AdminAccess,
) -> AdminQuestionResponse:
    if not isinstance(unanswered, PostgresUnansweredRepository):
        raise HTTPException(status_code=501, detail="Admin queue requires PostgreSQL")
    if answer_has_placeholders(submission.answer):
        raise HTTPException(
            status_code=422,
            detail="Replace every [[PLACEHOLDER]] before approving the answer",
        )
    item = unanswered.get_for_school(submission.school_id, unanswered_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Unanswered question not found")
    if item.status != "open":
        raise HTTPException(status_code=409, detail="Question is no longer open")
    knowledge_entry_id = None
    superseded_ids: list[UUID] = []
    if submission.publish_to_knowledge:
        if not isinstance(knowledge, PostgresKnowledgeRepository):
            raise HTTPException(status_code=501, detail="Knowledge publication requires PostgreSQL")
        if not submission.knowledge_title or len(submission.knowledge_title.strip()) < 2:
            raise HTTPException(status_code=422, detail="Knowledge title is required for publication")
        normalized_title = submission.knowledge_title.strip()
        if normalized_title.endswith("?"):
            raise HTTPException(
                status_code=422,
                detail="Use a reusable topic title, not the parent's question",
            )
        if (
            submission.valid_from
            and submission.expires_at
            and submission.expires_at <= submission.valid_from
        ):
            raise HTTPException(status_code=422, detail="Expiry must be after the valid-from date")
        knowledge_entry_id = knowledge.publish_reviewed_answer(
            submission.school_id,
            submission.knowledge_title,
            submission.answer,
            submission.valid_from,
            submission.expires_at,
        )
        superseded_ids = [
            entry_id
            for entry_id in submission.supersede_knowledge_ids
            if entry_id != knowledge_entry_id
        ]
    try:
        response = unanswered.save_answer(
            submission.school_id,
            unanswered_id,
            submission.answer,
            "approved",
            "pending" if item.contact_email and item.consent_to_contact else "not_applicable",
        )
        if knowledge_entry_id is not None:
            unanswered.link_resolution(
                submission.school_id,
                unanswered_id,
                knowledge_entry_id,
            )
            knowledge.archive_entries(submission.school_id, superseded_ids)
            response = response.model_copy(
                update={"knowledge_entry_id": knowledge_entry_id}
            )
        return await deliver_approved_answer(item, response)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/v1/admin/unanswered/{unanswered_id}/retry-email",
    response_model=AdminUnansweredQuestion,
)
async def retry_parent_email(
    unanswered_id: UUID,
    school_id: str,
    _: AdminAccess,
) -> AdminUnansweredQuestion:
    if not isinstance(unanswered, PostgresUnansweredRepository):
        raise HTTPException(status_code=501, detail="Admin queue requires PostgreSQL")
    item = unanswered.get_for_school(school_id, unanswered_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Unanswered question not found")
    if not item.contact_email or not item.consent_to_contact:
        raise HTTPException(status_code=409, detail="No consenting parent email is available")
    if item.response_status != "approved" or not item.latest_response_id or not item.latest_answer:
        raise HTTPException(status_code=409, detail="No approved answer is available")
    response = AdminQuestionResponse(
        id=item.latest_response_id,
        unanswered_id=item.id,
        answer=item.latest_answer,
        status="approved",
        created_at=item.responded_at or item.created_at,
        approved_at=item.responded_at,
        delivery_status=item.delivery_status or "failed",
        delivery_error=item.delivery_error,
        delivered_at=item.delivered_at,
    )
    await deliver_approved_answer(item, response)
    refreshed = unanswered.get_for_school(school_id, unanswered_id)
    if refreshed is None:
        raise HTTPException(status_code=404, detail="Unanswered question not found")
    return refreshed


@app.post("/v1/admin/knowledge", response_model=KnowledgeEntry, status_code=201)
async def create_knowledge(entry: KnowledgeEntry, _: AdminAccess) -> KnowledgeEntry:
    if not hasattr(knowledge, "add"):
        raise HTTPException(
            status_code=501,
            detail="PostgreSQL admin knowledge writes are not implemented yet",
        )
    return knowledge.add(entry)
