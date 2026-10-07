from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .embeddings import HashingEmbeddingModel, SentenceTransformerEmbeddingModel
from .llm import EvidenceOnlyModel, OllamaModel, VLLMModel
from .models import (
    AdminAnswerSubmission,
    AdminQuestionResponse,
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
from .service import ChatService, LeadRepository, UnansweredRepository


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
    try:
        return unanswered.save_answer(
            submission.school_id,
            unanswered_id,
            submission.answer,
            "approved",
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/v1/admin/knowledge", response_model=KnowledgeEntry, status_code=201)
async def create_knowledge(entry: KnowledgeEntry, _: AdminAccess) -> KnowledgeEntry:
    if not hasattr(knowledge, "add"):
        raise HTTPException(
            status_code=501,
            detail="PostgreSQL admin knowledge writes are not implemented yet",
        )
    return knowledge.add(entry)
