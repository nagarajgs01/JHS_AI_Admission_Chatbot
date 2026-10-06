from uuid import UUID, uuid4

from .llm import LanguageModel
from .models import ChatRequest, ChatResponse, Citation, Lead, LeadSubmission, UnansweredQuestion
from .retrieval import InMemoryKnowledgeRepository


ESCALATION_MESSAGE = (
    "I’m sorry, but I don’t have enough approved information to answer that confidently. "
    "No admissions agent is currently available. Please share your email address, and a "
    "school representative will contact you as soon as possible."
)


class UnansweredRepository:
    def __init__(self) -> None:
        self.items: dict[UUID, UnansweredQuestion] = {}

    def create(self, school_id: str, question: str) -> UnansweredQuestion:
        item = UnansweredQuestion(school_id=school_id, question=question)
        self.items[item.id] = item
        return item


class LeadRepository:
    def __init__(self) -> None:
        self.items: dict[UUID, Lead] = {}

    def create(self, submission: LeadSubmission) -> Lead:
        lead = Lead.model_validate(submission.model_dump())
        self.items[lead.id] = lead
        return lead


class ChatService:
    def __init__(
        self,
        knowledge: InMemoryKnowledgeRepository,
        unanswered: UnansweredRepository,
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
