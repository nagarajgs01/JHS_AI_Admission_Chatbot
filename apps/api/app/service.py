import re
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
    QueryUnderstanding,
    UnansweredQuestion,
)
from .retrieval import InMemoryKnowledgeRepository
from .retrieval_policy import (
    evaluate_evidence,
    expand_query,
    has_likely_typo,
    meaningful_keyword_score,
    synonym_evidence_score,
    typo_tolerant_query_tokens,
)


ESCALATION_MESSAGE = (
    "I’m sorry, but I don’t have enough approved information to answer that confidently. "
    "I’ve sent your question to the school team for review. You may add your email below "
    "if you would like to receive their answer directly."
)
CLARIFICATION_MESSAGE = "I’m not completely sure what you meant. Did you mean one of these?"
DETAILS_CLARIFICATION_MESSAGE = (
    "I can check that more accurately, but the answer depends on a few details. "
    "Please provide the information below."
)
OUT_OF_SCOPE_MESSAGE = (
    "I can help with Jain Heritage School admissions, academics, facilities, transport, "
    "fees, and related school enquiries. Please ask me a question about the school."
)

SOCIAL_RESPONSES = {
    "greeting": (
        "Hello! Welcome to Jain Heritage School Electronic City. How can I help you "
        "with the school, academics, facilities, transport, or admissions today?"
    ),
    "wellbeing": (
        "I’m doing well, thank you! I’m here to help with questions about Jain Heritage "
        "School Electronic City. What would you like to know?"
    ),
    "thanks": "You’re welcome! Please let me know if you have any other questions about the school.",
    "farewell": "Goodbye! Thank you for contacting Jain Heritage School Electronic City.",
}


def social_response(message: str) -> str | None:
    """Handle pure social turns locally without retrieval or escalation.

    These patterns intentionally match the complete message. A mixed message such
    as "Hi, what are the fees?" must continue through grounded school retrieval.
    """

    cleaned = re.sub(r"[^a-z0-9'\s]", " ", message.lower())
    cleaned = " ".join(cleaned.split())
    if re.fullmatch(
        r"(?:hi|hii+|hey|hello|hello there|hi there|good morning|good afternoon|"
        r"good evening|greetings)(?: (?:sir|madam|ma'am|team|there))?",
        cleaned,
    ):
        return SOCIAL_RESPONSES["greeting"]
    if re.fullmatch(
        r"(?:how are you|how are you doing|how is it going|how's it going|"
        r"hope you are well|hope you are doing well)",
        cleaned,
    ):
        return SOCIAL_RESPONSES["wellbeing"]
    if re.fullmatch(
        r"(?:thanks|thank you|thankyou|thanks a lot|thank you so much|"
        r"thanks for your help|that helps|got it|okay thanks|ok thanks)",
        cleaned,
    ):
        return SOCIAL_RESPONSES["thanks"]
    if re.fullmatch(
        r"(?:bye|goodbye|good bye|see you|see you later|have a good day|"
        r"have a nice day)",
        cleaned,
    ):
        return SOCIAL_RESPONSES["farewell"]
    return None

CANONICAL_QUESTIONS = {
    "school identity": "What is the school name?",
    "grades and curriculum": "What grades and curriculum does the school offer?",
    "boards offered": "Which boards are available?",
    "cbse pathway": "Tell me about the CBSE pathway.",
    "icse pathway": "Tell me about the ICSE pathway.",
    "early-years learning approach": "How do children learn in the early years?",
    "pre-primary daily rhythm": "What happens during a Pre-Primary day?",
    "finnish and nordic-inspired methodology": "What is the Finnish and Nordic-inspired methodology?",
    "learning and assessment in grades 1-7": "How are students in Grades 1–7 taught and assessed?",
    "educational philosophy": "What is the school’s educational philosophy?",
    "campus learning spaces": "What learning spaces are available on campus?",
    "sports facilities": "What sports facilities are available?",
    "library, yoga, dance and music": "Are library, yoga, dance and music available?",
    "school meal plan": "Does the school provide meals?",
    "transport and extended day care": "Does the school provide transport or extended day care?",
    "admission process": "What is the admission process?",
    "admission enquiry information": "What information is needed for an admission enquiry?",
    "pre-primary admission interaction": "How does Pre-Primary admission work?",
    "admission eligibility and age": "What are the admission age and eligibility requirements?",
    "fee information policy": "How can I get the current fee information?",
    "seat availability": "How can I check current seat availability?",
    "enquiry privacy statement": "How is admission enquiry information used?",
    "school contact details and location": "Where is the school and how can I contact it?",
    "jgi group background": "Tell me about the JGI Group.",
}


def canonical_question(title: str) -> str:
    return CANONICAL_QUESTIONS.get(title.strip().lower(), f"Tell me about {title.strip()}.")


def contextualize_question(question: str, context: dict[str, str]) -> str:
    clean_context = {
        key.strip(): value.strip()
        for key, value in context.items()
        if key.strip() and value.strip()
    }
    if not clean_context:
        return question
    details = "\n".join(
        f"- {key.replace('_', ' ')}: {value}" for key, value in clean_context.items()
    )
    return f"{question}\n\nADDITIONAL DETAILS\n{details}"


def is_conversation_followup(question: str, history) -> bool:
    """Detect references that need recent turns to become a standalone question."""

    if not history:
        return False
    lowered = " ".join(question.lower().split())
    token_count = len(re.findall(r"[a-z0-9]+", lowered))
    # Short questions commonly omit the subject established in the prior turn:
    # "How much does it cost?", "When is it available?", or "Who is the
    # teacher?". Six tokens covers these natural follow-ups while longer,
    # self-contained questions continue through the normal fast path.
    if token_count <= 6:
        return True
    if re.search(r"\b(it|its|that|this|those|these|they|them|there|same|former|latter)\b", lowered):
        return True
    # A definite noun phrase often points back to a grade, programme, route,
    # facility, or process already discussed in the session.
    if token_count <= 8 and re.search(
        r"\bthe\s+(teacher|fee|fees|cost|price|time|timing|timings|route|bus|"
        r"board|curriculum|syllabus|class|grade|campus|visit|process|application|"
        r"facility|facilities|programme|program)\b",
        lowered,
    ):
        return True
    return lowered.startswith(("and ", "also ", "then ", "what about ", "how about "))


def contextualize_with_history(question: str, history) -> str:
    """Add bounded, session-provided history for intent resolution only."""

    if not is_conversation_followup(question, history):
        return question
    recent = history[-6:]
    transcript = "\n".join(
        f"{item.role.upper()}: {item.content.strip()[:800]}" for item in recent
    )
    return (
        "RECENT CONVERSATION (context only; not approved evidence)\n"
        f"{transcript}\n\nCURRENT QUESTION\n{question}"
    )


def escalation_question(question: str, context: dict[str, str], history) -> str:
    """Persist enough context on an escalation ticket for a staff member to answer."""

    result = question.strip()
    # Staff should always receive the recent session snapshot. Even a seemingly
    # complete question may depend on a grade, location, or programme mentioned
    # earlier, and the transcript is context—not reusable knowledge.
    if history:
        transcript = "\n".join(
            f"- {item.role.title()}: {item.content.strip()[:800]}" for item in history[-6:]
        )
        result = f"{result}\n\nRECENT CONVERSATION\n{transcript}"
    clean_context = {
        key.strip(): value.strip()
        for key, value in context.items()
        if key.strip() and value.strip()
    }
    if clean_context:
        details = "\n".join(
            f"- {key.replace('_', ' ')}: {value}" for key, value in clean_context.items()
        )
        result = f"{result}\n\nADDITIONAL DETAILS\n{details}"
    return result


def is_clear_missing_information(question: str) -> bool:
    lowered = question.lower()
    terms = typo_tolerant_query_tokens(question)
    if {"teacher", "faculty"} & terms:
        return True
    if {
        "timing",
        "timings",
        "hours",
        "deadline",
        "seat",
        "seats",
        "vacancy",
        "availability",
    } & terms:
        return True
    if any(phrase in lowered for phrase in ("my child", "my application", "application status")):
        return True
    if any(marker in lowered for marker in ("current", "today", "right now", "tomorrow")) and (
        {"seat", "fee", "teacher", "timing", "deadline", "availability"} & terms
    ):
        return True
    return False


def candidate_matches_intent(question: str, candidate) -> bool:
    expanded_terms = typo_tolerant_query_tokens(expand_query(question))
    candidate_terms = typo_tolerant_query_tokens(
        f"{candidate.entry.title} {' '.join(candidate.entry.tags)}"
    )
    return bool(expanded_terms & candidate_terms)


def should_offer_clarification(question: str, candidates) -> bool:
    if not candidates or is_clear_missing_information(question):
        return False
    meaningful_terms = typo_tolerant_query_tokens(question)
    ambiguous_scores = (
        len(candidates) >= 2
        and abs(candidates[0].score - candidates[1].score) <= 0.06
    )
    return has_likely_typo(question) or len(meaningful_terms) <= 2 or ambiguous_scores


def is_obviously_out_of_scope(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    school_time_context = re.search(
        r"\b(school|class|admission|campus|application|deadline|birth|cutoff|cut-off|timing|timings|hours|visit|visiting|tour|appointment)\b",
        lowered,
    )
    if re.search(r"\btime\b", lowered) and not school_time_context:
        return True
    if (
        re.search(r"\bdate\b", lowered)
        and re.search(r"\b(today|todays|today's|current)\b", lowered)
        and not school_time_context
    ):
        return True
    patterns = (
        "what is the time now",
        "what time is it",
        "current time",
        "weather today",
        "today's weather",
        "tell me a joke",
        "latest news",
    )
    return any(pattern in lowered for pattern in patterns)


def needs_contextual_understanding(question: str) -> bool:
    """Reserve the LLM classifier for unclear or potentially personalized requests."""

    lowered = f" {' '.join(question.lower().split())} "
    personal_markers = (
        " my ",
        " me ",
        " our ",
        " mine ",
        " my child ",
        " this child ",
        " my area ",
        " near me ",
        " my location ",
        " my address ",
        " my application ",
    )
    return has_likely_typo(question) or any(marker in lowered for marker in personal_markers)


def detail_is_already_supplied(question: str, detail: str) -> bool:
    """Reject model-requested details whose value is already in the question."""

    normalized_detail = re.sub(r"[^a-z0-9]+", "_", detail.lower()).strip("_")
    lowered = " ".join(question.lower().split())
    if normalized_detail in {"grade", "class", "grade_or_class"}:
        return bool(
            re.search(
                r"\b(?:(?:grade|class)\s*(?:-|:)?\s*\d{1,2}|play\s*group|"
                r"playgroup|nursery|lkg|ukg)\b",
                lowered,
            )
        )
    if normalized_detail in {"email", "email_address"}:
        return bool(re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", question))
    if normalized_detail in {"phone", "mobile", "mobile_number", "contact_number"}:
        return bool(re.search(r"(?:\+?\d[\d\s()-]{7,}\d)", question))
    if normalized_detail in {"academic_year", "admission_year", "year"}:
        return bool(re.search(r"\b20\d{2}(?:\s*[-/]\s*\d{2,4})?\b", question))
    return False


def expanded_keyword_score(question: str, searchable_text: str) -> float:
    """Score original terms and controlled expansion terms without label noise."""

    original_score = meaningful_keyword_score(question, searchable_text)
    expanded = expand_query(question)
    marker = "Related search terms:"
    if marker not in expanded:
        return original_score
    expansion_terms = expanded.split(marker, 1)[1].strip()
    if not expansion_terms:
        return original_score
    return max(
        original_score,
        meaningful_keyword_score(expansion_terms, searchable_text),
    )


def has_strong_approved_hit(question: str, hits, minimum_score: float) -> bool:
    """Return whether deterministic retrieval already resolved the question."""

    if not hits or hits[0].score < minimum_score:
        return False
    top = hits[0]
    decision = evaluate_evidence(
        question,
        top.entry.content,
        top.score,
        minimum_score,
    )
    if not decision.supported:
        return False
    searchable_text = f"{top.entry.title} {top.entry.content}"
    lexical_coverage = expanded_keyword_score(question, searchable_text)
    synonym_match = synonym_evidence_score(question, searchable_text)
    return (
        top.score >= 0.50
        or lexical_coverage >= 0.60
        or synonym_match == 1.0
    )


def prioritize_complete_overview(question: str, hits, minimum_score: float):
    """Prefer a complete board overview for broad curriculum questions."""

    if not hits:
        return hits
    terms = typo_tolerant_query_tokens(question)
    if not terms & {"board", "curriculum", "syllabus"}:
        return hits
    requested_pathway = next(
        (pathway for pathway in ("cbse", "icse") if pathway in terms),
        None,
    )
    if requested_pathway:
        candidates = []
        other_pathway = "icse" if requested_pathway == "cbse" else "cbse"
        for index, hit in enumerate(hits):
            title = hit.entry.title.lower()
            if requested_pathway not in title or hit.score < minimum_score:
                continue
            decision = evaluate_evidence(
                question,
                hit.entry.content,
                hit.score,
                minimum_score,
            )
            if decision.supported:
                dedicated_pathway = f"{requested_pathway} pathway" in title
                excludes_other_pathway = other_pathway not in title
                candidates.append(
                    (
                        dedicated_pathway,
                        excludes_other_pathway,
                        hit.score,
                        index,
                        hit,
                    )
                )
        if candidates:
            _, _, _, index, hit = max(candidates)
            if index:
                return [hit, *hits[:index], *hits[index + 1:]]
            return hits
        return hits
    if terms & {"cisce", "ncert"}:
        return hits

    for index, hit in enumerate(hits):
        title = hit.entry.title.lower()
        content = hit.entry.content.lower()
        is_board_overview = (
            "board" in title
            and "cbse" in content
            and "icse" in content
        )
        if not is_board_overview or hit.score < minimum_score:
            continue
        decision = evaluate_evidence(
            question,
            hit.entry.content,
            hit.score,
            minimum_score,
        )
        if decision.supported and index:
            return [hit, *hits[:index], *hits[index + 1:]]
        if decision.supported:
            return hits
    return hits


def extractive_fast_answer(question: str, hits, minimum_score: float) -> str | None:
    """Return approved source text only when the match is exceptionally direct.

    This path performs no language-model generation, so it cannot invent a fact.
    It is intentionally conservative: the retrieval policy must approve the hit,
    and the hybrid score and question-to-evidence lexical coverage must both be
    strong. Titles are metadata and never override evidence contained in the
    approved passage. Ambiguous or synthesis-heavy questions keep using normal
    grounded generation and independent verification.
    """

    if not hits:
        return None
    top = hits[0]
    if top.score < minimum_score:
        return None
    content = top.entry.content.strip()
    decision = evaluate_evidence(
        question,
        content,
        top.score,
        minimum_score,
    )
    if not decision.supported:
        return None
    searchable_text = f"{top.entry.title} {top.entry.content}"
    lexical_coverage = expanded_keyword_score(question, searchable_text)
    synonym_match = synonym_evidence_score(question, searchable_text)
    # A high hybrid score already includes semantic, keyword and domain-synonym
    # evidence. Require exact/fuzzy lexical coverage only for borderline matches;
    # otherwise normal word forms such as transportation/transport and
    # offered/offers would trigger an unnecessary language-model round trip.
    deterministic_typo_match = has_likely_typo(question) and (
        top.score >= 0.50
        or lexical_coverage >= 0.60
        or synonym_match == 1.0
    )
    if (
        top.score < 0.55
        and lexical_coverage < 0.60
        and synonym_match < 1.0
        and not deterministic_typo_match
    ):
        return None
    if not content or len(content) > 700:
        return None
    return content

CALLBACK_SUGGESTION = (
    "Thank you for your enquiry. The school team is verifying the requested information "
    "and will respond once it has been confirmed."
)
PLACEHOLDER_RE = re.compile(r"\[\[[^\]]+\]\]")


def answer_has_placeholders(answer: str) -> bool:
    return bool(PLACEHOLDER_RE.search(answer))


def extract_question_context(question: str) -> dict[str, str]:
    parts = re.split(r"\n\s*ADDITIONAL DETAILS\s*\n", question, maxsplit=1, flags=re.I)
    if len(parts) < 2:
        return {}
    details: dict[str, str] = {}
    for line in parts[1].splitlines():
        cleaned = re.sub(r"^\s*-\s*", "", line).strip()
        key, separator, value = cleaned.partition(":")
        if separator and key.strip() and value.strip():
            details[re.sub(r"\s+", "_", key.strip().lower())] = value.strip()
    return details


def fill_template_context(template: str, question: str) -> str:
    """Fill only facts explicitly supplied by the parent; leave school facts blank."""

    context = extract_question_context(question)
    replacements = {
        "LOCATION OR ROUTE": context.get("pickup_area")
        or context.get("pickup_location")
        or context.get("location"),
        "GRADE OR CLASS": context.get("grade") or context.get("class"),
        "GRADE OR PROGRAM": context.get("grade") or context.get("program"),
        "ACADEMIC YEAR": context.get("academic_year") or context.get("admission_year"),
    }
    completed = template
    for placeholder, value in replacements.items():
        if value:
            completed = completed.replace(f"[[{placeholder}]]", value)
    return completed


def answer_templates(question: str) -> list[str]:
    """Return fact-free, intent-specific drafts with mandatory staff placeholders."""

    lowered = question.lower()
    if "teacher" in lowered or "faculty" in lowered:
        return [
            "The current [[GRADE OR CLASS]] class teacher for the [[ACADEMIC YEAR]] "
            "academic year is [[TEACHER NAME]]. For further assistance, please contact "
            "[[CONTACT PERSON OR DEPARTMENT]] at [[CONTACT DETAILS]]."
        ]
    if any(term in lowered for term in ("fee", "fees", "cost", "price")):
        return [
            "The [[FEE TYPE]] fee for [[GRADE OR PROGRAM]] for the [[ACADEMIC YEAR]] "
            "academic year is [[AMOUNT]]. This includes [[INCLUSIONS]] and excludes "
            "[[EXCLUSIONS]]. The payment schedule is [[PAYMENT SCHEDULE]]."
        ]
    if any(term in lowered for term in ("seat", "seats", "vacancy", "available")):
        return [
            "For the [[ACADEMIC YEAR]] academic year, seats for [[GRADE OR PROGRAM]] are "
            "currently [[AVAILABLE OR NOT AVAILABLE]]. The next step is [[NEXT ADMISSION STEP]]."
        ]
    if any(term in lowered for term in ("timing", "timings", "hours", "schedule")):
        return [
            "The school timings for [[GRADE OR PROGRAM]] are [[START TIME]] to [[END TIME]] "
            "on [[APPLICABLE DAYS]]. [[ADDITIONAL TIMING DETAILS OR EXCEPTIONS]]."
        ]
    if any(term in lowered for term in ("deadline", "last date", "closing date")):
        return [
            "The application deadline for [[GRADE OR PROGRAM]] for the [[ACADEMIC YEAR]] "
            "academic year is [[DEADLINE DATE]]. Applications can be submitted through "
            "[[APPLICATION METHOD OR LINK]]."
        ]
    if any(term in lowered for term in ("bus", "transport", "route")):
        return [
            "Transport for [[LOCATION OR ROUTE]] is [[AVAILABLE OR NOT AVAILABLE]]. The "
            "pickup point is [[PICKUP POINT]], the timing is [[PICKUP TIME]], and the fee is "
            "[[TRANSPORT FEE OR FEE POLICY]]."
        ]
    return [
        "Thank you for your question. [[DIRECT VERIFIED ANSWER]]. "
        "[[ADDITIONAL DETAILS, CONDITIONS, OR NEXT STEP]]."
    ]


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

    async def understand(self, question: str) -> QueryUnderstanding:
        try:
            understanding = await self.model.understand(question)
            if not understanding.normalized_question.strip():
                raise ValueError("Normalized question is empty")
            return understanding
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return QueryUnderstanding(
                normalized_question=question,
                scope="out_of_scope" if is_obviously_out_of_scope(question) else "school",
                clarity="ambiguous" if has_likely_typo(question) else "clear",
            )

    async def respond(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or uuid4()
        contextual_question = contextualize_question(request.message, request.context)
        followup = is_conversation_followup(contextual_question, request.history)
        understanding_question = contextualize_with_history(
            contextual_question,
            request.history,
        )
        ticket_question = escalation_question(
            request.message,
            request.context,
            request.history,
        )
        if request.force_escalation:
            unanswered = self.unanswered.create(request.school_id, ticket_question)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=ESCALATION_MESSAGE,
                outcome="escalated",
                confidence=0.0,
                unanswered_id=unanswered.id,
            )

        friendly_answer = social_response(request.message)
        if friendly_answer:
            return ChatResponse(
                conversation_id=conversation_id,
                answer=friendly_answer,
                outcome="answered",
                confidence=1.0,
            )

        # Interpret short continuations together with their recent turn. History
        # helps resolve intent, but it is never passed in as answer evidence.
        scope_question = understanding_question if followup else contextual_question
        if is_obviously_out_of_scope(scope_question):
            return ChatResponse(
                conversation_id=conversation_id,
                answer=OUT_OF_SCOPE_MESSAGE,
                outcome="out_of_scope",
                confidence=0.0,
            )

        # Try approved hybrid retrieval first. Clear, general questions do not need a
        # separate LLM classification call; this removes one full local-model round trip.
        hits = [] if followup else self.knowledge.search(request.school_id, contextual_question)
        hits = prioritize_complete_overview(
            contextual_question,
            hits,
            self.min_score,
        )
        deterministic_typo_match = (
            has_likely_typo(contextual_question)
            and has_strong_approved_hit(
                contextual_question,
                hits,
                self.min_score,
            )
        )
        if deterministic_typo_match or (
            not needs_contextual_understanding(contextual_question)
            and (hits or is_clear_missing_information(contextual_question))
        ):
            understanding = QueryUnderstanding(
                normalized_question=contextual_question,
                scope="school",
                clarity="clear",
            )
        else:
            understanding = await self.understand(understanding_question)
        if understanding.scope == "out_of_scope":
            return ChatResponse(
                conversation_id=conversation_id,
                answer=OUT_OF_SCOPE_MESSAGE,
                outcome="out_of_scope",
                confidence=0.0,
            )

        missing_details = [
            detail
            for detail in understanding.missing_details
            if detail
            and not request.context.get(detail, "").strip()
            and not detail_is_already_supplied(request.message, detail)
        ]
        if missing_details:
            return ChatResponse(
                conversation_id=conversation_id,
                answer=DETAILS_CLARIFICATION_MESSAGE,
                outcome="clarification",
                confidence=0.0,
                clarification_question=request.message,
                clarification_kind="details",
                required_details=missing_details[:4],
            )

        retrieval_parts = [understanding.normalized_question]
        if understanding.intent and understanding.intent != "general_school_enquiry":
            retrieval_parts.append(understanding.intent.replace("_", " "))
        retrieval_parts.extend(fact.replace("_", " ") for fact in understanding.requested_facts)
        retrieval_question = " ".join(dict.fromkeys(part for part in retrieval_parts if part))
        if retrieval_question != contextual_question or not hits:
            hits = self.knowledge.search(request.school_id, retrieval_question)
            hits = prioritize_complete_overview(
                retrieval_question,
                hits,
                self.min_score,
            )

        if not hits or (
            hits[0].score < self.min_score and not hits[0].evidence_supported
        ):
            candidates = (
                self.knowledge.clarification_candidates(
                    request.school_id,
                    retrieval_question,
                )
                if hasattr(self.knowledge, "clarification_candidates")
                else []
            )
            candidates = [
                candidate
                for candidate in candidates
                if candidate_matches_intent(retrieval_question, candidate)
            ]
            # A clear English question can still miss the strict answer threshold.
            # When a semantically related approved topic exists, offer that topic
            # instead of immediately creating an unanswered ticket.
            if candidates and not is_clear_missing_information(retrieval_question):
                options = list(
                    dict.fromkeys(canonical_question(hit.entry.title) for hit in candidates)
                )[:3]
                if options:
                    return ChatResponse(
                        conversation_id=conversation_id,
                        answer=CLARIFICATION_MESSAGE,
                        outcome="clarification",
                        confidence=candidates[0].score,
                        suggested_questions=options,
                        clarification_question=request.message,
                        clarification_kind="choice",
                    )
            unanswered = self.unanswered.create(request.school_id, ticket_question)
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
        answer_question = understanding.normalized_question if followup else contextual_question

        # Strong single-source matches are safest and fastest when returned
        # extractively.  The text is already published school knowledge, so this
        # path avoids both hallucination and two expensive local-model round trips.
        extractive_answer = extractive_fast_answer(
            answer_question,
            approved_hits,
            self.min_score,
        )
        if extractive_answer:
            hit = approved_hits[0]
            return ChatResponse(
                conversation_id=conversation_id,
                answer=extractive_answer,
                outcome="answered",
                confidence=hit.score,
                citations=[
                    Citation(
                        knowledge_id=hit.entry.id,
                        title=hit.entry.title,
                        source_label=hit.entry.source_label,
                        excerpt=hit.entry.content[:240],
                        score=hit.score,
                    )
                ],
            )

        answer = await self.model.answer(answer_question, approved_hits)
        if answer == "INSUFFICIENT_EVIDENCE":
            unanswered = self.unanswered.create(request.school_id, ticket_question)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=ESCALATION_MESSAGE,
                outcome="escalated",
                confidence=hits[0].score,
                unanswered_id=unanswered.id,
            )

        # Retrieval success does not make generated wording trustworthy. A separate,
        # constrained pass must verify every factual claim against the same evidence.
        if not await self.model.verify(answer_question, answer, approved_hits):
            unanswered = self.unanswered.create(request.school_id, ticket_question)
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
        try:
            understanding = await self.model.understand(question)
            retrieval_question = understanding.normalized_question
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            retrieval_question = question

        hits = self.knowledge.search(school_id, retrieval_question)
        if not hits and hasattr(self.knowledge, "clarification_candidates"):
            hits = self.knowledge.clarification_candidates(
                school_id,
                retrieval_question,
            )
        approved_hits = [
            hit
            for hit in hits
            if hit.score >= 0.18 or hit.evidence_supported
        ]

        suggestions: list[AdminAnswerSuggestion] = []
        if approved_hits:
            seen: set[str] = set()
            for hit in approved_hits[:3]:
                try:
                    candidates = await self.model.suggest(question, [hit])
                except (KeyError, RuntimeError, TypeError, ValueError):
                    candidates = []
                citation = Citation(
                    knowledge_id=hit.entry.id,
                    title=hit.entry.title,
                    source_label=hit.entry.source_label,
                    excerpt=hit.entry.content[:240],
                    score=hit.score,
                )
                for candidate in candidates:
                    normalized = " ".join(candidate.lower().split())
                    if normalized in seen:
                        existing = next(
                            (
                                item
                                for item in suggestions
                                if " ".join(item.answer.lower().split()) == normalized
                            ),
                            None,
                        )
                        if existing and all(
                            source.knowledge_id != citation.knowledge_id
                            for source in existing.citations
                        ):
                            existing.citations.append(citation)
                        continue
                    try:
                        verified = await self.model.verify(question, candidate, [hit])
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        verified = False
                    if verified:
                        seen.add(normalized)
                        suggestions.append(
                            AdminAnswerSuggestion(
                                answer=candidate,
                                kind="grounded",
                                citations=[citation],
                            )
                        )
                        if len(suggestions) >= 3:
                            break
                if len(suggestions) >= 3:
                    break

        for template in answer_templates(question):
            if len(suggestions) >= 3:
                break
            suggestions.append(
                AdminAnswerSuggestion(
                    answer=fill_template_context(template, question),
                    kind="template",
                    requires_staff_verification=True,
                )
            )

        if len(suggestions) < 3:
            suggestions.append(
                AdminAnswerSuggestion(
                    answer=CALLBACK_SUGGESTION,
                    kind="callback",
                    requires_staff_verification=True,
                )
            )

        return AdminSuggestionResponse(
            unanswered_id=unanswered_id,
            question=question,
            suggestions=suggestions,
        )
