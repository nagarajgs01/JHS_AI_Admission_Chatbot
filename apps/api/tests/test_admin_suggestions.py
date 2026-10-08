from uuid import uuid4

import pytest

from app.models import KnowledgeEntry, KnowledgeStatus
from app.retrieval import InMemoryKnowledgeRepository
from app.retrieval import SearchHit
from app.service import AdminSuggestionService, answer_has_placeholders, fill_template_context


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


class ConflictingEvidenceRepository:
    def __init__(self):
        self.entries = [
            KnowledgeEntry(
                school_id="school-a",
                title="School timings source A",
                content="The school day is from 8:30 AM to 3:30 PM.",
                source_label="Handbook A",
                status=KnowledgeStatus.PUBLISHED,
            ),
            KnowledgeEntry(
                school_id="school-a",
                title="School timings source B",
                content="The school day is from 9:00 AM to 3:00 PM.",
                source_label="Circular B",
                status=KnowledgeStatus.PUBLISHED,
            ),
        ]

    def search(self, school_id, question):
        del school_id, question
        return [SearchHit(entry=entry, score=0.6) for entry in self.entries]


class SourceAwareSuggestionModel(SuggestionModel):
    async def suggest(self, question, evidence):
        del question
        return [evidence[0].entry.content]

    async def verify(self, question, answer, evidence):
        del question
        return answer == evidence[0].entry.content


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
async def test_missing_evidence_returns_fill_in_template_without_invented_fact():
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
    template = next(item for item in result.suggestions if item.kind == "template")
    assert "[[TEACHER NAME]]" in template.answer
    assert "[[ACADEMIC YEAR]]" in template.answer
    assert answer_has_placeholders(template.answer)
    assert not answer_has_placeholders(
        "The current Grade 5 teacher for 2026-27 is Priya Sharma."
    )
    assert all(not item.citations for item in result.suggestions)


@pytest.mark.asyncio
async def test_conflicting_sources_are_returned_as_separate_grounded_options():
    service = AdminSuggestionService(
        ConflictingEvidenceRepository(),
        SourceAwareSuggestionModel(),
        min_score=0.38,
    )

    result = await service.generate(
        "school-a",
        uuid4(),
        "What are the school timings?",
    )

    grounded = [item for item in result.suggestions if item.kind == "grounded"]
    assert len(grounded) == 2
    assert grounded[0].citations[0].source_label == "Handbook A"
    assert grounded[1].citations[0].source_label == "Circular B"
    assert grounded[0].answer != grounded[1].answer


def test_parent_context_prefills_matching_template_fields_only():
    template = (
        "Transport for [[LOCATION OR ROUTE]] is [[AVAILABLE OR NOT AVAILABLE]]. "
        "The pickup time is [[PICKUP TIME]]."
    )
    question = (
        "Does the school bus come near me?\n\n"
        "ADDITIONAL DETAILS\n- pickup area: JP Nagar, 4th Block"
    )

    completed = fill_template_context(template, question)

    assert "Transport for JP Nagar, 4th Block" in completed
    assert "[[LOCATION OR ROUTE]]" not in completed
    assert "[[AVAILABLE OR NOT AVAILABLE]]" in completed
    assert "[[PICKUP TIME]]" in completed
