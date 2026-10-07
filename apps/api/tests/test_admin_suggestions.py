from uuid import uuid4

import pytest

from app.models import KnowledgeEntry, KnowledgeStatus
from app.retrieval import InMemoryKnowledgeRepository
from app.service import AdminSuggestionService


class SuggestionModel:
    async def suggest(self, question, evidence):
        del question, evidence
        return [
            "Bus transport is available within a 15 km radius.",
            "Bus transport is free and covers 50 km.",
        ]

    async def verify(self, question, answer, evidence):
        del question, evidence
        return "15 km" in answer


@pytest.mark.asyncio
async def test_only_verified_grounded_suggestions_are_returned():
    repository = InMemoryKnowledgeRepository(
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
    service = AdminSuggestionService(repository, SuggestionModel(), min_score=0.25)

    result = await service.generate(
        "school-a",
        uuid4(),
        "Is bus transport available?",
    )

    grounded = [item for item in result.suggestions if item.kind == "grounded"]
    assert len(grounded) == 1
    assert "15 km" in grounded[0].answer
    assert grounded[0].citations[0].source_label == "Transport policy"


@pytest.mark.asyncio
async def test_missing_evidence_returns_only_callback_drafts():
    service = AdminSuggestionService(
        InMemoryKnowledgeRepository([]),
        SuggestionModel(),
        min_score=0.25,
    )

    result = await service.generate(
        "school-a",
        uuid4(),
        "Who is the current Grade 5 teacher?",
    )

    assert result.suggestions
    assert all(item.kind == "callback" for item in result.suggestions)
    assert all(not item.citations for item in result.suggestions)
