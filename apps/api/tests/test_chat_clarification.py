from uuid import UUID

import pytest

from app.llm import EvidenceOnlyModel
from app.models import ChatRequest, KnowledgeEntry, KnowledgeStatus, QueryUnderstanding
from app.retrieval import SearchHit
from app.service import ChatService, UnansweredRepository


class CandidateOnlyRepository:
    def __init__(self):
        self.entry = KnowledgeEntry(
            school_id="school-a",
            title="Boards offered",
            content="The school publishes information about CBSE and ICSE pathways.",
            source_label="Approved curriculum page",
            status=KnowledgeStatus.PUBLISHED,
            tags=["board", "curriculum", "CBSE", "ICSE"],
        )

    def search(self, school_id, query):
        del school_id, query
        return []

    def clarification_candidates(self, school_id, query):
        del school_id, query
        return [SearchHit(entry=self.entry, score=0.34)]


class TransportCandidateRepository(CandidateOnlyRepository):
    def __init__(self):
        self.entry = KnowledgeEntry(
            school_id="school-a",
            title="Transport and extended day care",
            content="The school provides transport services based on route feasibility.",
            source_label="Approved campus information",
            status=KnowledgeStatus.PUBLISHED,
            tags=["transport", "bus", "route", "pickup", "drop-off"],
        )


class RoutingModel(EvidenceOnlyModel):
    async def understand(self, question):
        lowered = question.lower()
        if "time" in lowered or "todays date" in lowered:
            return QueryUnderstanding(
                normalized_question=question,
                scope="out_of_scope",
                clarity="clear",
                intent="current_date_or_time",
            )
        if "wat bord" in lowered:
            return QueryUnderstanding(
                normalized_question="Which board does the school follow?",
                scope="school",
                clarity="ambiguous",
                intent="board_options",
                requested_facts=["board_options"],
            )
        return QueryUnderstanding(
            normalized_question=question,
            scope="school",
            clarity="clear",
        )


class ContextAwareRoutingModel(EvidenceOnlyModel):
    async def understand(self, question):
        personalized = "my area" in question.lower()
        has_location = "additional details" in question.lower()
        return QueryUnderstanding(
            normalized_question=question,
            scope="school",
            clarity="clear",
            intent="personalized_transport_coverage" if personalized else "transport_availability",
            requested_facts=["route_coverage"] if personalized else ["transport_availability"],
            missing_details=["pickup_area"] if personalized and not has_location else [],
        )


@pytest.mark.asyncio
async def test_unclear_typo_returns_clickable_clarification_without_creating_ticket():
    unanswered = UnansweredRepository()
    service = ChatService(
        CandidateOnlyRepository(),
        unanswered,
        RoutingModel(),
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(school_id="school-a", message="wat bord skul folow")
    )

    assert response.outcome == "clarification"
    assert response.suggested_questions == ["Which boards are available?"]
    assert response.unanswered_id is None
    assert not unanswered.items


@pytest.mark.asyncio
async def test_clear_paraphrase_offers_semantic_topic_instead_of_escalating():
    unanswered = UnansweredRepository()
    service = ChatService(
        TransportCandidateRepository(),
        unanswered,
        RoutingModel(),
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(school_id="school-a", message="What transportation choices are there?")
    )

    assert response.outcome == "clarification"
    assert response.suggested_questions == [
        "Does the school provide transport or extended day care?"
    ]
    assert response.unanswered_id is None
    assert not unanswered.items


@pytest.mark.asyncio
async def test_clear_missing_teacher_fact_escalates_instead_of_clarifying():
    unanswered = UnansweredRepository()
    service = ChatService(
        CandidateOnlyRepository(),
        unanswered,
        RoutingModel(),
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(
            school_id="school-a",
            message="Who is the current Grade 5 class teacher?",
        )
    )

    assert response.outcome == "escalated"
    assert isinstance(response.unanswered_id, UUID)
    assert not response.suggested_questions


@pytest.mark.asyncio
async def test_clear_school_timings_question_rejects_unrelated_candidates():
    unanswered = UnansweredRepository()
    service = ChatService(
        CandidateOnlyRepository(),
        unanswered,
        RoutingModel(),
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(school_id="school-a", message="What are the school timings?")
    )

    assert response.outcome == "escalated"
    assert response.unanswered_id is not None
    assert not response.suggested_questions


@pytest.mark.asyncio
async def test_personalized_question_collects_missing_context_before_creating_ticket():
    unanswered = UnansweredRepository()
    service = ChatService(
        CandidateOnlyRepository(),
        unanswered,
        ContextAwareRoutingModel(),
        min_score=0.38,
    )

    first_response = await service.respond(
        ChatRequest(school_id="school-a", message="Does the school bus come to my area?")
    )

    assert first_response.outcome == "clarification"
    assert first_response.clarification_kind == "details"
    assert first_response.required_details == ["pickup_area"]
    assert not unanswered.items

    second_response = await service.respond(
        ChatRequest(
            school_id="school-a",
            message="Does the school bus come to my area?",
            conversation_id=first_response.conversation_id,
            context={"pickup_area": "HSR Layout, Bengaluru"},
        )
    )

    assert second_response.outcome == "escalated"
    ticket = next(iter(unanswered.items.values()))
    assert "pickup area: HSR Layout, Bengaluru" in ticket.question


@pytest.mark.asyncio
async def test_general_transport_question_does_not_request_personal_details():
    service = ChatService(
        CandidateOnlyRepository(),
        UnansweredRepository(),
        ContextAwareRoutingModel(),
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(school_id="school-a", message="Does the school provide transport?")
    )

    assert response.clarification_kind != "details"
    assert response.required_details == []


@pytest.mark.asyncio
async def test_none_of_these_forces_escalation():
    unanswered = UnansweredRepository()
    service = ChatService(
        CandidateOnlyRepository(),
        unanswered,
        RoutingModel(),
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(
            school_id="school-a",
            message="wat bord skul folow",
            force_escalation=True,
        )
    )

    assert response.outcome == "escalated"
    assert response.unanswered_id is not None
    assert len(unanswered.items) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "What is the time now?",
        "Whats the time?",
        "Whats todays date?",
    ],
)
async def test_obvious_out_of_scope_question_does_not_create_school_ticket(question):
    unanswered = UnansweredRepository()
    service = ChatService(
        CandidateOnlyRepository(),
        unanswered,
        RoutingModel(),
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(school_id="school-a", message=question)
    )

    assert response.outcome == "out_of_scope"
    assert response.unanswered_id is None
    assert not unanswered.items
