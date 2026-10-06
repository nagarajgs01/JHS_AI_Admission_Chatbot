from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .embeddings import HashingEmbeddingModel, SentenceTransformerEmbeddingModel
from .llm import EvidenceOnlyModel, OllamaModel, VLLMModel
from .models import (
    ChatRequest,
    ChatResponse,
    EscalationContact,
    KnowledgeEntry,
    Lead,
    LeadSubmission,
)
from .postgres_knowledge import PostgresKnowledgeRepository
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
    allow_headers=["Content-Type", "Authorization"],
)


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
    item = unanswered.items.get(contact.unanswered_id)
    if item is None or item.school_id != contact.school_id:
        raise HTTPException(status_code=404, detail="Unanswered question not found")
    if not contact.consent_to_contact:
        raise HTTPException(status_code=422, detail="Consent is required")
    item.contact_email = contact.email
    item.consent_to_contact = True


@app.post("/v1/leads", response_model=Lead, status_code=status.HTTP_201_CREATED)
async def create_lead(submission: LeadSubmission) -> Lead:
    return leads.create(submission)


@app.post("/v1/admin/knowledge", response_model=KnowledgeEntry, status_code=201)
async def create_knowledge(entry: KnowledgeEntry) -> KnowledgeEntry:
    # Authentication/RBAC is the next milestone; this route is local-development only.
    if not hasattr(knowledge, "add"):
        raise HTTPException(
            status_code=501,
            detail="PostgreSQL admin knowledge writes are not implemented yet",
        )
    return knowledge.add(entry)
