from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class KnowledgeStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class KnowledgeEntry(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    school_id: str
    title: str
    content: str
    source_label: str
    source_url: str | None = None
    status: KnowledgeStatus = KnowledgeStatus.DRAFT
    tags: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    expires_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Citation(BaseModel):
    knowledge_id: UUID
    title: str
    source_label: str
    excerpt: str
    score: float


class ChatRequest(BaseModel):
    school_id: str = Field(min_length=2, max_length=80)
    message: str = Field(min_length=2, max_length=2000)
    conversation_id: UUID | None = None
    force_escalation: bool = False
    context: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    outcome: Literal["answered", "clarification", "escalated", "out_of_scope"]
    confidence: float
    citations: list[Citation] = Field(default_factory=list)
    unanswered_id: UUID | None = None
    suggested_questions: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    clarification_kind: Literal["choice", "details"] | None = None
    required_details: list[str] = Field(default_factory=list)


class QueryUnderstanding(BaseModel):
    normalized_question: str
    scope: Literal["school", "out_of_scope"]
    clarity: Literal["clear", "ambiguous"]
    intent: str = "general_school_enquiry"
    requested_facts: list[str] = Field(default_factory=list)
    missing_details: list[str] = Field(default_factory=list)


class EscalationContact(BaseModel):
    school_id: str
    unanswered_id: UUID
    email: str = Field(min_length=5, max_length=254)
    consent_to_contact: bool

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        local, separator, domain = value.strip().partition("@")
        if not separator or not local or "." not in domain:
            raise ValueError("A valid email address is required")
        return value.strip().lower()


class UnansweredQuestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    school_id: str
    question: str
    contact_email: str | None = None
    consent_to_contact: bool = False
    status: str = "open"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdminUnansweredQuestion(UnansweredQuestion):
    latest_response_id: UUID | None = None
    latest_answer: str | None = None
    response_status: Literal["draft", "approved"] | None = None
    responded_at: datetime | None = None
    delivery_status: Literal["pending", "sent", "failed", "not_applicable"] | None = None
    delivery_error: str | None = None
    delivered_at: datetime | None = None


class AdminAnswerSubmission(BaseModel):
    school_id: str = Field(min_length=2, max_length=80)
    answer: str = Field(min_length=2, max_length=5000)
    publish_to_knowledge: bool = False
    knowledge_title: str | None = Field(default=None, max_length=200)
    valid_from: datetime | None = None
    expires_at: datetime | None = None
    supersede_knowledge_ids: list[UUID] = Field(default_factory=list)


class AdminQuestionResponse(BaseModel):
    id: UUID
    unanswered_id: UUID
    answer: str
    status: Literal["draft", "approved"]
    created_at: datetime
    approved_at: datetime | None = None
    knowledge_entry_id: UUID | None = None
    delivery_status: Literal["pending", "sent", "failed", "not_applicable"] = "not_applicable"
    delivery_error: str | None = None
    delivered_at: datetime | None = None


class AdminAnswerSuggestion(BaseModel):
    answer: str
    kind: Literal["grounded", "template", "callback"]
    requires_staff_verification: bool = True
    citations: list[Citation] = Field(default_factory=list)


class AdminSuggestionResponse(BaseModel):
    unanswered_id: UUID
    question: str
    suggestions: list[AdminAnswerSuggestion]


class LeadSubmission(BaseModel):
    school_id: str
    student_name: str = Field(min_length=2, max_length=120)
    grade_applied_for: str = Field(min_length=1, max_length=40)
    guardian_name: str = Field(min_length=2, max_length=120)
    mobile_number: str = Field(min_length=7, max_length=20)
    email: str = Field(min_length=5, max_length=254)
    preferred_contact: str | None = Field(default=None, max_length=160)
    consent_to_contact: bool

    @field_validator("email")
    @classmethod
    def validate_lead_email(cls, value: str) -> str:
        local, separator, domain = value.strip().partition("@")
        if not separator or not local or "." not in domain:
            raise ValueError("A valid email address is required")
        return value.strip().lower()

    @field_validator("mobile_number")
    @classmethod
    def validate_mobile(cls, value: str) -> str:
        cleaned = "".join(character for character in value if character.isdigit() or character == "+")
        digits = "".join(character for character in cleaned if character.isdigit())
        if not 7 <= len(digits) <= 15:
            raise ValueError("A valid mobile number is required")
        return cleaned

    @field_validator("consent_to_contact")
    @classmethod
    def require_consent(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Consent is required")
        return value


class Lead(LeadSubmission):
    id: UUID = Field(default_factory=uuid4)
    status: str = "new"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
