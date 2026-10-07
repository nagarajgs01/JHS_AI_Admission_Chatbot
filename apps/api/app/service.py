from typing import Protocol
from uuid import UUID, uuid4

from .llm import LanguageModel
from .models import (
    AdminAnswerSuggestion,
    AdminSuggestionResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    Lead,
    LeadSubmission,
    UnansweredQuestion,
)
from .retrieval import InMemoryKnowledgeRepository


ESCALATION_MESSAGE = (
    "I’m sorry, but I don’t have enough approved information to answer that confidently. "
    "No admissions agent is currently available. Please share your email address, and a "
    "school representative will contact you as soon as possible."
)

CALLBACK_SUGGESTIONS = (
    "Thank you for your enquiry. The requested information is not available in the "
    "currently approved school information. The school team will verify it and contact you.",
    "Thank you for contacting the school. We have forwarded your question to the appropriate "
    "team and will respond after the information has been confirmed.",
)


class UnansweredRepository:
    def __init__(self) -> None:
        self.items: dict[UUID, UnansweredQuestion] = {}

    def create(self, school_id: str, question: str) -> UnansweredQuestion:
        item = UnansweredQuestion(school_id=school_id, question=question)
        self.items[item.id] = item
        return item

    def get(self, unanswered_id: UUID) -> UnansweredQuestion | None:
        return self.items.get(unanswered_id)

    def attach_contact(self, unanswered_id: UUID, email: str) -> None:
        item = self.items[unanswered_id]
        item.contact_email = email
        item.consent_to_contact = True


class LeadRepository:
    def __init__(self) -> None:
        self.items: dict[UUID, Lead] = {}

    def create(self, submission: LeadSubmission) -> Lead:
        lead = Lead.model_validate(submission.model_dump())
        self.items[lead.id] = lead
        return lead


class UnansweredStore(Protocol):
    def create(self, school_id: str, question: str) -> UnansweredQuestion: ...
    def get(self, unanswered_id: UUID) -> UnansweredQuestion | None: ...
    def attach_contact(self, unanswered_id: UUID, email: str) -> None: ...


class LeadStore(Protocol):
    def create(self, submission: LeadSubmission) -> Lead: ...


class ChatService:
    def __init__(
        self,
        knowledge: InMemoryKnowledgeRepository,
        unanswered: UnansweredStore,
        model: LanguageModel,
        min_score: float,
    ) -> None:
        self.knowledge = knowledge
        self.unanswered = unanswered
        self.model = model
        self.min_score = min_score

    async def respond(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or uuid4()
        hits = self.knowledge.search(request.school_id, request.message)

        if not hits or (
            hits[0].score < self.min_score and not hits[0].evidence_supported
        ):
            unanswered = self.unanswered.create(request.school_id, request.message)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=ESCALATION_MESSAGE,
                outcome="escalated",
                confidence=hits[0].score if hits else 0.0,
                unanswered_id=unanswered.id,
            )

        # Do not send weak, merely-nearby passages to the language model.
        # This also keeps irrelevant citations out of the client response.
        approved_hits = [
            hit
            for hit in hits
            if hit.score >= self.min_score or hit.evidence_supported
        ]
        answer = await self.model.answer(request.message, approved_hits)
        if answer == "INSUFFICIENT_EVIDENCE":
            unanswered = self.unanswered.create(request.school_id, request.message)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=ESCALATION_MESSAGE,
                outcome="escalated",
                confidence=hits[0].score,
                unanswered_id=unanswered.id,
            )

        # Retrieval success does not make generated wording trustworthy. A separate,
        # constrained pass must verify every factual claim against the same evidence.
        if not await self.model.verify(request.message, answer, approved_hits):
            unanswered = self.unanswered.create(request.school_id, request.message)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=ESCALATION_MESSAGE,
                outcome="escalated",
                confidence=hits[0].score,
                unanswered_id=unanswered.id,
            )

        citations = [
            Citation(
                knowledge_id=hit.entry.id,
                title=hit.entry.title,
                source_label=hit.entry.source_label,
                excerpt=hit.entry.content[:240],
                score=hit.score,
            )
            for hit in approved_hits
        ]
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            outcome="answered",
            confidence=hits[0].score,
            citations=citations,
        )


class AdminSuggestionService:
    def __init__(self, knowledge, model: LanguageModel, min_score: float) -> None:
        self.knowledge = knowledge
        self.model = model
        self.min_score = min_score

    async def generate(
        self,
        school_id: str,
        unanswered_id: UUID,
        question: str,
    ) -> AdminSuggestionResponse:
        hits = self.knowledge.search(school_id, question)
        approved_hits = [
            hit
            for hit in hits
            if hit.score >= self.min_score or hit.evidence_supported
        ]

        suggestions: list[AdminAnswerSuggestion] = []
        if approved_hits:
            try:
                candidates = await self.model.suggest(question, approved_hits)
            except (KeyError, RuntimeError, TypeError, ValueError):
                candidates = []

            citations = [
                Citation(
                    knowledge_id=hit.entry.id,
                    title=hit.entry.title,
                    source_label=hit.entry.source_label,
                    excerpt=hit.entry.content[:240],
                    score=hit.score,
                )
                for hit in approved_hits
            ]
            seen: set[str] = set()
            for candidate in candidates:
                normalized = " ".join(candidate.lower().split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                try:
                    verified = await self.model.verify(
                        question,
                        candidate,
                        approved_hits,
                    )
                except (KeyError, RuntimeError, TypeError, ValueError):
                    verified = False
                if verified:
                    suggestions.append(
                        AdminAnswerSuggestion(
                            answer=candidate,
                            kind="grounded",
                            citations=citations,
                        )
                    )

        for callback in CALLBACK_SUGGESTIONS:
            if len(suggestions) >= 3:
                break
            suggestions.append(
                AdminAnswerSuggestion(
                    answer=callback,
                    kind="callback",
                    requires_staff_verification=True,
                )
            )

        return AdminSuggestionResponse(
            unanswered_id=unanswered_id,
            question=question,
            suggestions=suggestions,
        )
