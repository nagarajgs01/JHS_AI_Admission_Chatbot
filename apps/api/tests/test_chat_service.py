from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from app.llm import EvidenceOnlyModel
from app.models import (
    ChatHistoryMessage,
    ChatRequest,
    KnowledgeEntry,
    KnowledgeStatus,
    QueryUnderstanding,
)
from app.retrieval import InMemoryKnowledgeRepository
from app.service import (
    ChatService,
    ESCALATION_MESSAGE,
    UnansweredRepository,
    contextualize_with_history,
    escalation_question,
    is_conversation_followup,
    social_response,
)
from app.retrieval_policy import has_likely_typo


class HallucinatingModel:
    async def answer(self, question, evidence):
        del question, evidence
        return "Bus transport is free and covers 50 km."

    async def verify(self, question, answer, evidence):
        del question, answer, evidence
        return False

    async def health(self):
        return True


class CountingEvidenceModel(EvidenceOnlyModel):
    def __init__(self):
        self.understanding_calls = 0

    async def understand(self, question):
        self.understanding_calls += 1
        return await super().understand(question)


class FollowupEvidenceModel(EvidenceOnlyModel):
    def __init__(self, normalized_question: str):
        self.normalized_question = normalized_question
        self.understanding_input = ""

    async def understand(self, question):
        self.understanding_input = question
        return QueryUnderstanding(
            normalized_question=self.normalized_question,
            scope="school",
            clarity="clear",
            intent="transport",
            requested_facts=["transport"],
        )


def test_normal_word_forms_do_not_trigger_typo_understanding_path():
    assert not has_likely_typo("What are the transportation options?")
    assert not has_likely_typo("What fees apply to different grades?")


@pytest.mark.parametrize(
    ("message", "expected_text"),
    [
        ("Hello!", "Welcome"),
        ("How are you?", "doing well"),
        ("Thank you", "welcome"),
        ("Goodbye", "Goodbye"),
    ],
)
def test_pure_social_messages_get_friendly_responses(message, expected_text):
    assert expected_text in social_response(message)


def test_mixed_greeting_and_school_question_is_not_consumed_as_small_talk():
    assert social_response("Hi, what are the school fees?") is None


@pytest.mark.asyncio
async def test_social_message_does_not_create_an_unanswered_ticket():
    unanswered = UnansweredRepository()
    service = ChatService(
        InMemoryKnowledgeRepository([]), unanswered, EvidenceOnlyModel(), min_score=0.38
    )
    response = await service.respond(
        ChatRequest(school_id="school-a", message="Thank you!")
    )
    assert response.outcome == "answered"
    assert response.citations == []
    assert response.unanswered_id is None
    assert unanswered.items == {}


def test_short_followup_uses_only_session_history():
    history = [
        ChatHistoryMessage(role="user", content="Does the school provide transport?"),
        ChatHistoryMessage(role="assistant", content="Yes, subject to route feasibility."),
    ]
    assert is_conversation_followup("What about the fee?", history)
    prompt = contextualize_with_history("What about the fee?", history)
    assert "Does the school provide transport?" in prompt
    assert "CURRENT QUESTION\nWhat about the fee?" in prompt


def test_natural_short_cost_question_is_a_followup():
    history = [
        ChatHistoryMessage(role="user", content="Do you have transportation?"),
        ChatHistoryMessage(role="assistant", content="Yes, transport is available."),
    ]
    assert is_conversation_followup("How much does it costs?", history)
    prompt = contextualize_with_history("How much does it costs?", history)
    assert "Do you have transportation?" in prompt


def test_escalated_followup_keeps_context_for_school_staff():
    history = [
        ChatHistoryMessage(role="user", content="Does the school provide transport?"),
        ChatHistoryMessage(role="assistant", content="Transport depends on the route."),
    ]
    stored = escalation_question("What about my area?", {"pickup_area": "JP Nagar"}, history)
    assert stored.startswith("What about my area?")
    assert "RECENT CONVERSATION" in stored
    assert "Does the school provide transport?" in stored
    assert "pickup area: JP Nagar" in stored


@pytest.mark.asyncio
async def test_followup_is_rewritten_then_grounded_in_approved_evidence():
    repo = InMemoryKnowledgeRepository(
        [
            KnowledgeEntry(
                school_id="school-a",
                title="Transport fee policy",
                content="Transport fees depend on the route and pickup location.",
                source_label="Approved transport policy",
                status=KnowledgeStatus.PUBLISHED,
                tags=["transport", "bus", "fee", "route"],
            )
        ]
    )
    model = FollowupEvidenceModel("How is the school transport fee calculated?")
    service = ChatService(repo, UnansweredRepository(), model, min_score=0.1)
    history = [
        ChatHistoryMessage(role="user", content="Does the school provide buses?"),
        ChatHistoryMessage(role="assistant", content="Yes, transport is available."),
    ]
    response = await service.respond(
        ChatRequest(
            school_id="school-a", message="What about the fee?", history=history
        )
    )
    assert response.outcome == "answered"
    assert "Transport fees depend" in response.answer
    assert "RECENT CONVERSATION" in model.understanding_input
    assert response.citations[0].source_label == "Approved transport policy"


@pytest.mark.asyncio
async def test_unanswerable_followup_ticket_contains_recent_conversation():
    model = FollowupEvidenceModel("Who is the current Grade 5 class teacher?")
    unanswered = UnansweredRepository()
    service = ChatService(InMemoryKnowledgeRepository([]), unanswered, model, min_score=0.38)
    history = [
        ChatHistoryMessage(role="user", content="My child is applying to Grade 5."),
        ChatHistoryMessage(role="assistant", content="Grade 5 applications are supported."),
    ]
    response = await service.respond(
        ChatRequest(
            school_id="school-a", message="Who is the teacher?", history=history
        )
    )
    assert response.outcome == "escalated"
    stored = unanswered.get(response.unanswered_id)
    assert stored is not None
    assert "My child is applying to Grade 5." in stored.question


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
async def test_clear_general_question_uses_fast_path_and_contextual_synonyms():
    repo = InMemoryKnowledgeRepository(
        [
            KnowledgeEntry(
                school_id="school-a",
                title="Transport and extended day care",
                content="The school offers transport services based on route feasibility.",
                source_label="Approved campus information",
                status=KnowledgeStatus.PUBLISHED,
                tags=["transport", "bus", "route"],
            )
        ]
    )
    model = CountingEvidenceModel()
    service = ChatService(repo, UnansweredRepository(), model, min_score=0.38)

    response = await service.respond(
        ChatRequest(school_id="school-a", message="What are the transportation options?")
    )

    assert response.outcome == "answered"
    assert "transport services" in response.answer
    assert model.understanding_calls == 0


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
