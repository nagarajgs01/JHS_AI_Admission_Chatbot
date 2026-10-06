from uuid import UUID

import pytest

from app.llm import EvidenceOnlyModel
from app.models import ChatRequest, KnowledgeEntry, KnowledgeStatus
from app.retrieval import InMemoryKnowledgeRepository
from app.service import ChatService, ESCALATION_MESSAGE, UnansweredRepository


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
