from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from app.llm import EvidenceOnlyModel
from app.models import ChatRequest, KnowledgeEntry, KnowledgeStatus
from app.retrieval import InMemoryKnowledgeRepository
from app.service import ChatService, ESCALATION_MESSAGE, UnansweredRepository


class HallucinatingModel:
    async def answer(self, question, evidence):
        del question, evidence
        return "Bus transport is free and covers 50 km."

    async def verify(self, question, answer, evidence):
        del question, answer, evidence
        return False

    async def health(self):
        return True


@pytest.mark.asyncio
async def test_answers_from_published_same_school_content():
    repo = InMemoryKnowledgeRepository(
        [
            KnowledgeEntry(
                school_id="school-a",
                title="Transport availability",
                content="Bus transport is available within a 15 km radius.",
                source_label="Transport policy",
                status=KnowledgeStatus.PUBLISHED,
                tags=["bus", "transport"],
            )
        ]
    )
    service = ChatService(repo, UnansweredRepository(), EvidenceOnlyModel(), min_score=0.25)
    response = await service.respond(ChatRequest(school_id="school-a", message="Is bus transport available?"))

    assert response.outcome == "answered"
    assert "15 km" in response.answer
    assert response.citations[0].source_label == "Transport policy"


@pytest.mark.asyncio
async def test_draft_and_other_school_content_are_never_used():
    repo = InMemoryKnowledgeRepository(
        [
            KnowledgeEntry(
                school_id="school-b",
                title="Fees",
                content="Private fee information.",
                source_label="Draft",
                status=KnowledgeStatus.DRAFT,
            )
        ]
    )
    unanswered = UnansweredRepository()
    service = ChatService(repo, unanswered, EvidenceOnlyModel(), min_score=0.1)
    response = await service.respond(ChatRequest(school_id="school-a", message="What are the fees?"))

    assert response.outcome == "escalated"
    assert response.answer == ESCALATION_MESSAGE
    assert isinstance(response.unanswered_id, UUID)
    assert len(unanswered.items) == 1


@pytest.mark.asyncio
async def test_unrelated_shared_words_do_not_create_false_answer():
    repo = InMemoryKnowledgeRepository(
        [
            KnowledgeEntry(
                school_id="school-a",
                title="Meal plan",
                content="The school serves balanced meals.",
                source_label="Brochure",
                status=KnowledgeStatus.PUBLISHED,
            )
        ]
    )
    service = ChatService(repo, UnansweredRepository(), EvidenceOnlyModel(), min_score=0.58)
    response = await service.respond(ChatRequest(school_id="school-a", message="What are the school fees?"))

    assert response.outcome == "escalated"
    assert response.citations == []


@pytest.mark.asyncio
async def test_unverified_generated_answer_is_never_returned():
    repo = InMemoryKnowledgeRepository(
        [
            KnowledgeEntry(
                school_id="school-a",
                title="Transport availability",
                content="Bus transport is available within a 15 km radius.",
                source_label="Transport policy",
                status=KnowledgeStatus.PUBLISHED,
                tags=["bus", "transport"],
            )
        ]
    )
    unanswered = UnansweredRepository()
    service = ChatService(repo, unanswered, HallucinatingModel(), min_score=0.25)

    response = await service.respond(
        ChatRequest(school_id="school-a", message="Is bus transport available?")
    )

    assert response.outcome == "escalated"
    assert response.answer == ESCALATION_MESSAGE
    assert response.citations == []
    assert response.unanswered_id is not None


@pytest.mark.asyncio
async def test_expired_knowledge_is_never_returned():
    repo = InMemoryKnowledgeRepository(
        [
            KnowledgeEntry(
                school_id="school-a",
                title="Admission deadline",
                content="The admission deadline is January 10.",
                source_label="Expired notice",
                status=KnowledgeStatus.PUBLISHED,
                tags=["admission", "deadline"],
                expires_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
        ]
    )
    service = ChatService(repo, UnansweredRepository(), EvidenceOnlyModel(), min_score=0.1)

    response = await service.respond(
        ChatRequest(school_id="school-a", message="What is the admission deadline?")
    )

    assert response.outcome == "escalated"
    assert response.citations == []
