from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
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
    detail_is_already_supplied,
    deterministic_missing_details,
    escalation_question,
    expanded_keyword_score,
    extractive_fast_answer,
    has_strong_approved_hit,
    is_conversation_followup,
    prioritize_complete_overview,
    social_response,
)
from app.retrieval_policy import correct_query_typos, expand_query, has_likely_typo


class HallucinatingModel:
    def __init__(self):
        self.answer_calls = 0
        self.verify_calls = 0

    async def answer(self, question, evidence):
        self.answer_calls += 1
        del question, evidence
        return "Bus transport is free and covers 50 km."

    async def verify(self, question, answer, evidence):
        self.verify_calls += 1
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


class RedundantMissingGradeModel(EvidenceOnlyModel):
    async def understand(self, question):
        del question
        return QueryUnderstanding(
            normalized_question="Who is the current Grade 5 class teacher?",
            scope="school",
            clarity="needs_details",
            intent="staff_information",
            requested_facts=["teacher"],
            missing_details=["grade"],
        )


def test_normal_word_forms_do_not_trigger_typo_understanding_path():
    assert not has_likely_typo("What are the transportation options?")
    assert not has_likely_typo("What fees apply to different grades?")
    assert not has_likely_typo("What boards are offered?")
    assert not has_likely_typo("Which classes use school buses?")
    assert has_likely_typo("What syllabuss do you offer?")


@pytest.mark.parametrize(
    ("question", "corrected"),
    [
        ("What syllabuss do you offer?", "What syllabus do you offer?"),
        ("Do u provide transpotation?", "Do you provide transportation?"),
        ("Hw can I aply?", "How can I apply?"),
        ("Who is the chairmn?", "Who is the chairman?"),
        ("Do you hav swiming facilites?", "Do you have swimming facilities?"),
    ],
)
def test_confident_domain_typos_are_corrected_before_expansion(question, corrected):
    assert correct_query_typos(question) == corrected


def test_corrected_terms_receive_domain_expansion():
    transport_query = expand_query("Do u provide transpotation?")
    assert transport_query.startswith("Do you provide transportation?")
    assert "bus" in transport_query
    assert "route" in transport_query

    application_query = expand_query("Hw can I aply?")
    assert application_query.startswith("How can I apply?")
    assert "admission process" in application_query


def test_structured_detail_values_are_detected_in_the_question():
    assert detail_is_already_supplied(
        "Who is the current Grade 5 class teacher?", "grade"
    )
    assert detail_is_already_supplied("Is Nursery available?", "grade")
    assert not detail_is_already_supplied("Which grades are available?", "grade")


def test_transport_location_dependency_is_detected_without_model():
    assert deterministic_missing_details(
        "Does the school bus come to my area?", {}
    ) == ["pickup_area"]
    assert deterministic_missing_details(
        "Does the school bus come to my area?", {"pickup_area": "JP Nagar"}
    ) == []
    assert deterministic_missing_details("Does the school provide transport?", {}) == []


@pytest.mark.asyncio
async def test_model_cannot_request_a_grade_already_present_in_question():
    unanswered = UnansweredRepository()
    service = ChatService(
        InMemoryKnowledgeRepository([]),
        unanswered,
        RedundantMissingGradeModel(),
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(
            school_id="school-a",
            message="Who is the current Grade 5 class teacher?",
        )
    )

    assert response.outcome == "escalated"
    assert response.required_details == []
    assert response.unanswered_id is not None


@pytest.mark.asyncio
async def test_clear_dynamic_question_skips_llm_understanding_before_escalation():
    model = CountingEvidenceModel()
    service = ChatService(
        InMemoryKnowledgeRepository([]),
        UnansweredRepository(),
        model,
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(
            school_id="school-a",
            message="Who is the current Grade 5 class teacher?",
        )
    )

    assert response.outcome == "escalated"
    assert model.understanding_calls == 0


@pytest.mark.asyncio
async def test_transport_location_clarification_skips_llm_understanding():
    model = CountingEvidenceModel()
    service = ChatService(
        InMemoryKnowledgeRepository([]),
        UnansweredRepository(),
        model,
        min_score=0.38,
    )

    response = await service.respond(
        ChatRequest(
            school_id="school-a",
            message="Does the school bus come to my area?",
        )
    )

    assert response.outcome == "clarification"
    assert response.required_details == ["pickup_area"]
    assert model.understanding_calls == 0


def test_approved_meal_match_uses_fast_path_at_half_score():
    entry = KnowledgeEntry(
        school_id="school-a",
        title="School meal plan",
        content="The school provides meals cooked on campus.",
        source_label="Approved campus page",
        status=KnowledgeStatus.PUBLISHED,
        tags=["meal", "food"],
    )
    hits = [SimpleNamespace(entry=entry, score=0.5234)]

    assert extractive_fast_answer(
        "Does the school provide meals?", hits, 0.38
    ) == entry.content


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
async def test_strong_direct_evidence_bypasses_generation_safely():
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
    model = HallucinatingModel()
    service = ChatService(repo, unanswered, model, min_score=0.25)

    response = await service.respond(
        ChatRequest(school_id="school-a", message="Is bus transport available?")
    )

    assert response.outcome == "answered"
    assert response.answer == "Bus transport is available within a 15 km radius."
    assert response.citations[0].source_label == "Transport policy"
    assert response.unanswered_id is None
    assert model.answer_calls == 0
    assert model.verify_calls == 0


def test_close_runner_up_does_not_override_strong_approved_evidence():
    identity = KnowledgeEntry(
        school_id="school-a",
        title="School identity",
        content=(
            "The official name is Jain Heritage School Electronic City, also "
            "referred to as JHS Electronic City."
        ),
        source_label="Approved school website",
        status=KnowledgeStatus.PUBLISHED,
        tags=["school", "identity"],
    )
    nearby = KnowledgeEntry(
        school_id="school-a",
        title="Admission enquiry information",
        content="Families provide the child's name when making an admission enquiry.",
        source_label="Approved admissions page",
        status=KnowledgeStatus.PUBLISHED,
        tags=["admissions", "enquiry"],
    )
    hits = [
        SimpleNamespace(entry=identity, score=0.5385),
        SimpleNamespace(entry=nearby, score=0.4854),
    ]

    answer = extractive_fast_answer("What is the school name?", hits, 0.38)

    assert answer == identity.content


@pytest.mark.parametrize(
    ("question", "title", "content", "score"),
    [
        (
            "Do you provide bus transportation?",
            "Transport and extended day care",
            "The school states that school transport with pick-up and drop is available.",
            0.6405,
        ),
        (
            "What boards are offered?",
            "Boards offered",
            "The Electronic City campus offers both CBSE and ICSE.",
            0.7365,
        ),
    ],
)
def test_strong_hybrid_match_handles_equivalent_word_forms(
    question, title, content, score
):
    entry = KnowledgeEntry(
        school_id="school-a",
        title=title,
        content=content,
        source_label="Approved school website",
        status=KnowledgeStatus.PUBLISHED,
        tags=[],
    )
    hits = [SimpleNamespace(entry=entry, score=score)]

    assert extractive_fast_answer(question, hits, 0.38) == content


def test_policy_approved_typo_match_skips_language_model_route():
    entry = KnowledgeEntry(
        school_id="school-a",
        title="Boards offered",
        content="The Electronic City campus offers both CBSE and ICSE.",
        source_label="Approved school website",
        status=KnowledgeStatus.PUBLISHED,
        tags=["board", "curriculum"],
    )
    hits = [SimpleNamespace(entry=entry, score=0.5364)]
    question = "What syllabuss do you offer?"

    assert has_strong_approved_hit(question, hits, 0.38)
    assert extractive_fast_answer(question, hits, 0.38) == entry.content


def test_expanded_query_match_skips_language_model_route():
    entry = KnowledgeEntry(
        school_id="school-a",
        title="Admission process",
        content=(
            "The admission process starts with an enquiry and campus visit, "
            "followed by an application and enrolment."
        ),
        source_label="Approved admissions page",
        status=KnowledgeStatus.PUBLISHED,
        tags=["admission", "application"],
    )
    hits = [
        SimpleNamespace(entry=entry, score=0.4519)
    ]
    question = "Hw can I aply?"

    assert expanded_keyword_score(
        question, f"{entry.title} {entry.content}"
    ) >= 0.60
    assert has_strong_approved_hit(question, hits, 0.38)
    assert extractive_fast_answer(question, hits, 0.38) == entry.content


def test_broad_syllabus_question_prefers_complete_board_overview():
    icse = KnowledgeEntry(
        school_id="school-a",
        title="ICSE pathway",
        content="The ICSE pathway follows the CISCE curriculum.",
        source_label="Approved ICSE page",
        status=KnowledgeStatus.PUBLISHED,
        tags=["icse", "curriculum"],
    )
    overview = KnowledgeEntry(
        school_id="school-a",
        title="Boards offered",
        content="The school offers both CBSE and ICSE pathways.",
        source_label="Approved admissions page",
        status=KnowledgeStatus.PUBLISHED,
        tags=["board", "cbse", "icse"],
    )
    comparison = KnowledgeEntry(
        school_id="school-a",
        title="Choosing between CBSE and ICSE",
        content="The school helps families compare CBSE and ICSE.",
        source_label="Approved comparison page",
        status=KnowledgeStatus.PUBLISHED,
        tags=["cbse", "icse", "board"],
    )
    hits = [
        SimpleNamespace(entry=comparison, score=0.7177),
        SimpleNamespace(entry=icse, score=0.6153),
        SimpleNamespace(entry=overview, score=0.5558),
    ]

    ranked = prioritize_complete_overview(
        "What syllabus does the school follow?", hits, 0.38
    )

    assert ranked[0].entry.title == "Boards offered"
    assert prioritize_complete_overview(
        "Explain the ICSE syllabus", hits, 0.38
    )[0].entry.title == "ICSE pathway"


def test_explicit_cbse_question_prefers_cbse_pathway():
    overview = KnowledgeEntry(
        school_id="school-a",
        title="Boards offered",
        content="The school offers both CBSE and ICSE pathways.",
        source_label="Approved admissions page",
        status=KnowledgeStatus.PUBLISHED,
        tags=["board", "cbse", "icse"],
    )
    cbse = KnowledgeEntry(
        school_id="school-a",
        title="CBSE pathway",
        content="The CBSE pathway is NCERT-aligned and follows the NEP framework.",
        source_label="Approved CBSE page",
        status=KnowledgeStatus.PUBLISHED,
        tags=["cbse", "ncert"],
    )
    hits = [
        SimpleNamespace(entry=overview, score=0.7118),
        SimpleNamespace(entry=cbse, score=0.6000),
    ]

    ranked = prioritize_complete_overview("Explain the CBSE syllabus", hits, 0.38)

    assert ranked[0].entry.title == "CBSE pathway"


@pytest.mark.asyncio
async def test_unverified_generated_answer_is_never_returned():
    # A long source is deliberately ineligible for the concise extractive path,
    # forcing this request through generation and independent verification.
    approved_content = (
        "Bus transport is available within a 15 km radius. "
        + "Route availability remains subject to the approved transport policy. " * 12
    )
    repo = InMemoryKnowledgeRepository(
        [
            KnowledgeEntry(
                school_id="school-a",
                title="Transport availability",
                content=approved_content,
                source_label="Transport policy",
                status=KnowledgeStatus.PUBLISHED,
                tags=["bus", "transport"],
            )
        ]
    )
    unanswered = UnansweredRepository()
    model = HallucinatingModel()
    service = ChatService(repo, unanswered, model, min_score=0.25)

    response = await service.respond(
        ChatRequest(school_id="school-a", message="Is bus transport available?")
    )

    assert response.outcome == "escalated"
    assert response.answer == ESCALATION_MESSAGE
    assert response.citations == []
    assert response.unanswered_id is not None
    assert model.answer_calls == 1
    assert model.verify_calls == 1


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
