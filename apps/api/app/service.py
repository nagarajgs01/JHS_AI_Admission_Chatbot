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
    correct_query_typos,
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
    greeting = (
        r"(?:hi|hii+|hey|hello|hello there|hi there|good morning|good afternoon|"
        r"good evening|greetings)(?: (?:sir|madam|ma'am|team|there))?"
    )
    wellbeing = (
        r"(?:how are you|how are you doing|how is it going|how's it going|"
        r"hope you are well|hope you are doing well)"
    )

    # Treat combinations made entirely from social phrases as social turns too.
    # This remains deliberately full-message matched, so a school question such
    # as "Hello, how are you, and what are the fees?" is not swallowed here.
    if re.fullmatch(
        rf"(?:{greeting})(?: (?:and )?(?:{wellbeing}))",
        cleaned,
    ):
        return SOCIAL_RESPONSES["wellbeing"]
    if re.fullmatch(
        greeting,
        cleaned,
    ):
        return SOCIAL_RESPONSES["greeting"]
    if re.fullmatch(
        wellbeing,
        cleaned,
    ):
        return SOCIAL_RESPONSES["wellbeing"]
    if re.fullmatch(
        r"(?:thanks|thank you|thankyou|thanks a lot|thank you so much|"
        r"thanks for your help|thank you for your help|thank you for helping me|"
        r"thanks for helping me|thank you for the information|that helps|got it|"
        r"okay thanks|ok thanks)",
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
    if re.search(r"\b(it|its|that|this|those|these|they|them|there|same|former|latter|her|him)\b", lowered):
        return True
    if re.search(
        r"\b(?:which one|which option|how much|what cost|who provides|"
        r"what time can (?:we|i)|when can (?:we|i)|how does that|"
        r"how\s+(?:to\s+)?(?:book|schedule|arrange)|"
        r"what time (?:is good|should (?:we|i)|can (?:we|i))|"
        r"what subjects?|tell me more|give me (?:more )?(?:details|information)|"
        r"what (?:abilities|skills|benefits)|"
        r"(?:explain|describe).*(?:simply|more)|"
        r"one of (?:them|those)|what about)\b",
        lowered,
    ):
        return True
    # A definite noun phrase often points back to a grade, programme, route,
    # facility, or process already discussed in the session.
    if token_count <= 8 and re.search(
        r"\bthe\s+((?:class\s+)?teacher|fee|fees|cost|price|time|timing|timings|route|bus|"
        r"board|curriculum|syllabus|class|grade|campus|visit|process|application|"
        r"facility|facilities|programme|program)\b",
        lowered,
    ):
        return True
    return lowered.startswith(("and ", "also ", "then ", "what about ", "how about "))


def followup_retrieval_query(question: str, history) -> str:
    """Combine recent user turns for retrieval without treating them as evidence."""

    if not is_conversation_followup(question, history):
        return question
    recent_user_turns = [
        item.content.strip()
        for item in history
        if item.role == "user" and item.content.strip()
    ][-2:]
    return " ".join([*recent_user_turns, question.strip()])


def resolve_contextual_question(question: str, history) -> str:
    """Turn a session follow-up into a focused standalone retrieval question.

    Conversation text is used only to resolve references. It never becomes answer
    evidence; the final answer must still come from a published knowledge entry.
    """

    current = " ".join(question.strip().split())
    lowered = current.lower()

    if re.search(
        r"\b(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(?:school|office|campus)\s+"
        r"(?:time|timing|timings|hours)\b|"
        r"\bwhat\s+time\s+does\s+(?:the\s+)?(?:school|office|campus)\s+open\b|"
        r"\bwhat\s+time\s+(?:can|should)\s+(?:i|we)\s+"
        r"(?:contact|call|phone|reach)\s+(?:the\s+)?school\b|"
        r"\b(?:school|office|campus)\s+(?:opening\s+)?(?:time|timing|timings|hours)\b",
        lowered,
    ):
        return "What are the published school office and campus opening hours?"

    if re.search(r"\b(?:is|are)\b.*\b(?:united states|usa|u\.s\.a\.?|u\.s\.?)\b", lowered):
        return "Is Jain Heritage School Electronic City in the United States or in Bengaluru, Karnataka?"

    if re.search(
        r"\b(?:contact|reach|call|phone|telephone|email|e-mail)\b",
        lowered,
    ) or re.search(
        r"\b(?:talk|speak)\s+to\s+(?:(?:the\s+)?a\s+)?(?:person|someone|staff|team)\b|"
        r"\b(?:want|what)\s+to\s+(?:talk|speak)\s+to\s+(?:(?:the\s+)?a\s+)?"
        r"(?:person|someone|staff|team)\b",
        lowered,
    ):
        return "What are the school's phone number and email contact details?"

    # Canonicalize clear intent shapes before considering conversation history.
    # These queries use the vocabulary of the approved knowledge topics so that
    # generic wording does not rank an incidental passage above the intended one.
    if re.search(r"\b(?:assess|assessment|evaluate|evaluation)\w*\b", lowered):
        return (
            "How are students in Grades 1-7 assessed through portfolios, "
            "projects, presentations and unit tests?"
        )
    if re.search(
        r"\b(?:may|can|could|would)\b.*\b(?:come|see|tour|visit)\b.*\bcampus\b|"
        r"\b(?:tour|visit)\s+(?:the\s+)?campus\b",
        lowered,
    ):
        return "Can parents visit the campus, and should the visit be booked in advance?"
    if re.search(r"\b(?:apply|application)\b", lowered) and re.search(
        r"\b(?:start|begin|how|first|initial)\b", lowered
    ):
        return (
            "How can parents start the admission process by enquiring, visiting "
            "the campus and completing an application?"
        )
    if re.search(r"\bhow\b.*\b(?:book|schedule|arrange|make)\w*\b", lowered) and re.search(
        r"\b(?:visit|campus|appointment)\b", lowered
    ):
        return "How can I arrange a campus visit by submitting an enquiry or calling admissions?"

    if not history:
        return current
    user_turns = [
        item.content.strip()
        for item in history[-8:]
        if str(item.role).lower().endswith("user") and item.content.strip()
    ]
    prior = " ".join(user_turns).lower()

    if (
        re.search(r"\bhow\b.*\b(?:book|schedule|arrange|make)\w*\b", lowered)
        or re.search(r"\bhow\s+(?:do|can)\s+(?:i|we)\s+do\s+(?:it|so|that)\b", lowered)
        or re.search(r"\bhow\s+to\s+do\s+(?:it|so|that)\b", lowered)
    ) and re.search(r"\b(?:visit|campus|appointment|book|booking)\b", prior):
        return "How can I arrange a campus visit by submitting an enquiry or calling admissions?"

    if re.search(
        r"\bwhat\s+days?\b.*\bopen\b|\bwhich\s+days?\b.*\bopen\b|"
        r"\bdays?\s+(?:is|are)\s+(?:it|the\s+(?:school|office))\s+open\b",
        lowered,
    ) and re.search(
        r"\b(?:office|school|campus)\b.*\b(?:hours?|time|timings?|open)\b",
        prior,
    ):
        return "On which days is the school office open, and what are its office hours?"

    if re.search(r"\bwhat\s+time\s+(?:is\s+)?(?:good|best|suitable)\b", lowered) and re.search(
        r"\b(?:contact|call|phone|reach)\b.*\bschool\b|"
        r"\bschool\b.*\b(?:contact|call|phone|reach)\b",
        prior,
    ):
        return "What are the published school office and campus opening hours?"

    if re.search(r"\b(?:why|purpose|reason).*(?:interaction)|conduct.*interaction\b", lowered) and re.search(
        r"\b(?:pre-primary|nursery|entrance|admission interaction)\b", prior
    ):
        return (
            "Why does the school conduct the Pre-Primary admission interaction, "
            "and how does it help understand how the child learns?"
        )

    if re.search(r"\b(?:assess|assessment|evaluate|evaluation|exam|examination|rank|ranking)\w*\b", lowered):
        if re.search(r"\b(?:rank|ranking|ranked|against each other)\b", lowered):
            return "Does the school publish a rank order for children?"
        return current

    if re.search(r"\b(?:food[- ]?service|meal[- ]?service).*(?:partner|provider)|who.*(?:partner|provider)\b", lowered):
        return "Who is the school meal-service partner?"
    if re.search(r"\b(?:allerg|dietary|intolerance|special diet)\w*\b", lowered) and re.search(r"\b(?:meal|food)\b", prior):
        return "How does the school handle food allergies and dietary needs?"

    if re.search(r"\b(?:after|step after|what comes next).*(?:form|application)\b", lowered):
        return "What happens after completing and submitting the admission application?"

    if re.search(r"\b(?:booking|book|appointment)\b", lowered) and re.search(r"\b(?:campus|visit|tour|come|see)\b", prior):
        return "Should a campus visit be booked in advance?"

    if re.search(
        r"\b(?:neighbou?rhoods?|localit(?:y|ies)|areas?)\b.*"
        r"\b(?:cover|covered|serve|served)\b",
        lowered,
    ) and re.search(r"\b(?:bus(?:es)?|transport(?:ation)?|route)\b", prior):
        return "Does school transport cover every area, or must route coverage be confirmed?"

    if re.search(r"\bwhy\b.*\bonly\s+(?:cbse|icse)\b", lowered):
        return "Does the school offer both CBSE and ICSE?"

    # An explicitly named school entity is stronger than the surrounding
    # conversation and should become the retrieval anchor.
    named_pathway = next(
        (pathway.upper() for pathway in ("cbse", "icse") if pathway in lowered),
        None,
    )
    if named_pathway and re.search(
        r"\b(?:tell|explain|describe|details?|information|more|pathway|programme|program|curriculum|syllabus)\b",
        lowered,
    ):
        return f"Explain the {named_pathway} pathway and curriculum."

    # Carry the most recently named board into a pronoun-based exam follow-up.
    if re.search(r"\b(?:jee|neet|cuet|exam|exams|entrance examination)\b", lowered):
        recent_board = None
        for turn in reversed(user_turns):
            match = re.search(r"\b(cbse|icse)\b", turn, re.I)
            if match:
                recent_board = match.group(1).upper()
                break
        if recent_board:
            return f"Does the {recent_board} pathway align with JEE, NEET or CUET preparation?"

    if re.search(r"\bwhere(?:\s+exactly)?\s+(?:is|are)\s+(?:it|that|the school)\b", lowered):
        return "Where is Jain Heritage School Electronic City located?"

    if re.search(r"\b(?:what|which)\s+subjects?\b", lowered):
        if "icse" in prior:
            return "What subjects does the ICSE curriculum include?"
        if "cbse" in prior:
            return "What subjects does the CBSE curriculum include?"

    if re.search(r"\b(?:book|booking|appointment|reserve)\b", lowered) and re.search(
        r"\b(?:come|coming|visit|campus|before)\b", lowered
    ):
        return "Should a campus visit be booked in advance?"

    grade = re.findall(r"\b(?:grade|class)\s*(\d{1,2})\b", prior, re.I)
    if grade and re.search(r"\b(?:age|old|birth|dob|eligib)\w*\b", lowered):
        return f"What is the exact age requirement for Grade {grade[-1]}?"

    if re.search(r"\b(?:which|what)\s+(?:board|curriculum|one).*\b(?:suit|fit|better)\b|\bmay suit\b", lowered):
        preferences = []
        for term in ("languages", "language", "writing", "science", "mathematics", "maths", "move between cities", "portable"):
            if term in prior:
                preferences.append(term)
        suffix = f" The child prefers {' and '.join(dict.fromkeys(preferences))}." if preferences else ""
        return f"Which board may suit the child?{suffix}"

    return followup_retrieval_query(question, history)


def context_acknowledgement(message: str) -> str | None:
    """Acknowledge parent context statements without creating a staff ticket."""

    cleaned = " ".join(message.strip().split())
    lowered = cleaned.lower()
    if "?" in cleaned or re.search(
        r"^(?:please\s+)?(?:tell me|explain|i want to know|i need to know|"
        r"what|which|who|when|where|why|how|does|do|is|are|can|could|will)\b",
        lowered,
    ):
        return None
    family_context = re.search(r"\b(?:my|our)\s+(?:child|son|daughter)\b", lowered)
    application_context = re.search(
        r"\b(?:applying|application|interested|considering|seeking admission|"
        r"will join|will be in|joining|plans? to join|enjoys?|likes?|prefers?)\b",
        lowered,
    )
    if family_context and application_context:
        return (
            "Thank you for sharing that context. What would you like to know about "
            "your child’s application?"
        )
    return None


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
        "holiday",
        "holidays",
    } & terms:
        return True
    if re.search(r"\b(?:today|tomorrow|next|current|this week|this month)\b", lowered) and re.search(
        r"\b(?:menu|meal plan|lunch|holiday|calendar|event|meeting|schedule)\b",
        lowered,
    ):
        return True
    if any(phrase in lowered for phrase in ("my child", "my application", "application status")):
        return True
    if any(marker in lowered for marker in ("current", "today", "right now", "tomorrow")) and (
        {"seat", "fee", "teacher", "timing", "deadline", "availability"} & terms
    ):
        return True
    return False


def is_named_staff_question(question: str) -> bool:
    """Return whether the request asks for the identity of teaching staff."""

    lowered = " ".join(question.lower().split())
    terms = typo_tolerant_query_tokens(question)
    return bool(
        re.search(r"\b(?:who|name|named)\b", lowered)
        and (
            {"teacher", "faculty"} & terms
            or re.search(
                r"\b(?:teach|teaches|teaching|taught|instructor|instructors)\b",
                lowered,
            )
        )
    )


def is_school_authority_response(hit) -> bool:
    return (
        hit.entry.source_label.strip().lower()
        == "school-authority approved response"
    )


def mentioned_grades(value: str) -> set[str]:
    """Return normalized grade numbers explicitly stated in text."""

    lowered = " ".join(value.lower().split())
    grades = set(
        re.findall(
            r"\b(?:grade|class)\s*(\d{1,2})(?:st|nd|rd|th)?\b",
            lowered,
        )
    )
    grades.update(
        re.findall(
            r"\b(\d{1,2})(?:st|nd|rd|th)\s+(?:grade|class)\b",
            lowered,
        )
    )
    return grades


def staff_hit_matches_requested_grade(question: str, hit) -> bool:
    """Prevent an approved staff assignment from leaking across grades."""

    requested = mentioned_grades(question)
    if not requested:
        # A generic "Who is the teacher?" request does not identify which
        # grade-specific assignment the parent means. Never pick one merely
        # because it is the closest available approved record.
        return False
    evidence = mentioned_grades(f"{hit.entry.title} {hit.entry.content}")
    return bool(evidence and requested & evidence)


def most_recent_user_grades(history) -> set[str]:
    """Return grades from the newest user turn that explicitly mentions one."""

    for item in reversed(history or []):
        if not str(item.role).lower().endswith("user"):
            continue
        grades = mentioned_grades(item.content)
        if grades:
            return grades
    return set()


def grades_from_context(context: dict[str, str]) -> set[str]:
    """Normalize grade values submitted through the clarification form."""

    grades: set[str] = set()
    for key in ("grade", "class", "grade_or_class"):
        value = (context.get(key) or "").strip()
        if not value:
            continue
        grades.update(mentioned_grades(f"Grade {value}"))
    return grades


def is_dynamic_operational_question(question: str) -> bool:
    """Identify live operational facts that static approved knowledge cannot prove."""

    lowered = " ".join(question.lower().split())
    time_sensitive = re.search(
        r"\b(?:today|tomorrow|tonight|next|current|right now|this week|this month|"
        r"this term|this year)\b",
        lowered,
    )
    operational = re.search(
        r"\b(?:menu|food|meal|lunch|holiday|calendar|parent-teacher meeting|ptm|event|"
        r"class teacher|seat availability|seats available)\b",
        lowered,
    )
    return bool(time_sensitive and operational)


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
    arithmetic = re.fullmatch(
        r"(?:what\s+is|what's|calculate|compute|solve)?\s*"
        r"[-+]?\d+(?:\.\d+)?\s*(?:\+|-|\*|x|×|/|÷)\s*"
        r"[-+]?\d+(?:\.\d+)?\s*[?=]?",
        lowered,
    )
    arithmetic_words = re.fullmatch(
        r"(?:what\s+is|what's|calculate|compute|solve)\s+"
        r"[-+]?\d+(?:\.\d+)?\s+"
        r"(?:plus|minus|times|multiplied\s+by|divided\s+by)\s+"
        r"[-+]?\d+(?:\.\d+)?\s*\??",
        lowered,
    )
    if arithmetic or arithmetic_words:
        return True
    if re.search(
        r"\b(?:write|create|generate|debug|fix)\b.*\b(?:python|javascript|java|"
        r"typescript|code|program|function|array|sql)\b",
        lowered,
    ):
        return True
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
    if re.search(r"\b(today|tonight|yesterday|tomorrow|latest|current)\b", lowered) and re.search(
        r"\b(won|winner|score|match|game|news|weather|temperature)\b",
        lowered,
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


def is_prompt_injection(question: str) -> bool:
    """Reject requests to bypass grounding or fabricate school information."""

    lowered = " ".join(question.lower().split())
    bypass = re.search(
        r"\b(?:ignore|disregard|override|forget|bypass)\b.*"
        r"\b(?:instruction|instructions|rules|policy|prompt|system|"
        r"knowledge base|sources?|evidence)\b",
        lowered,
    )
    fabrication = re.search(
        r"\b(?:invent|fabricate|make up|guess)\b.*"
        r"\b(?:answer|details?|fees?|facts?|information|deadline|seat|"
        r"schedule|timings?|name|date)\b",
        lowered,
    )
    return bool(bypass or fabrication)


def contains_unsupported_negative_premise(question: str) -> bool:
    """Detect a question that asks the bot to accept an unverified negative claim."""

    lowered = " ".join(question.lower().split())
    negative_group_claim = bool(
        re.search(r"\bwhy\s+(?:does|do|is|are|can|will)\b", lowered)
        and re.search(
            r"\b(?:not|never|refuse|refuses|reject|rejects|ban|bans|forbid|forbids|"
            r"exclude|excludes|deny|denies|doesn't|don't|isn't|aren't)\b",
            lowered,
        )
        and (
            re.search(
                r"\b(?:girls?|boys?|religion|caste|disability|disabled|"
                r"nationality|race|gender)\b",
                lowered,
            )
            or re.search(
                r"\b(?:allow|accept|admit|reject|deny|exclude)\b.*"
                r"\b(?:children|students|applicants|families)\b",
                lowered,
            )
        )
    )
    compulsory_claim = bool(
        re.search(r"\bwhy\s+(?:is|are)\b", lowered)
        and re.search(r"\b(?:compulsory|mandatory|required)\b", lowered)
    )
    return negative_group_claim or compulsory_claim


def supplied_pickup_area(question: str) -> str | None:
    """Extract a parent-supplied route locality without treating it as school fact."""

    explicit_place = re.search(
        r"\b(?:from|near|around|at)\s+([a-z0-9][a-z0-9 .'-]{1,60}?)(?:[?.!,]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if explicit_place:
        value = " ".join(explicit_place.group(1).strip().split())
        if value.lower() not in {"me", "my area", "my location", "my address", "there", "here"}:
            return value
    match = re.search(
        r"\b(?:from|at|in|near|around|to|cover|covers|serve|serves|collect|collects|pick up|pickup|pick-up)\s+"
        r"([a-z0-9][a-z0-9 .'-]{1,60}?)(?:[?.!,]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = " ".join(match.group(1).strip().split())
    if value.lower() in {
        "me", "my area", "my location", "my address", "our area",
        "our location", "there", "here", "the area", "this area",
    }:
        return None
    return value


def transport_location_continuation(question: str, history) -> str | None:
    """Return a locality supplied after the bot requested a pickup area."""

    if not history:
        return None
    cleaned = " ".join(question.strip().split()).strip(".,?!")
    if not cleaned or len(cleaned) > 80 or len(cleaned.split()) > 8:
        return None
    if re.search(
        r"\b(?:what|which|who|when|where|why|how|does|do|is|are|can|could|will)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return None
    recent_user = " ".join(
        item.content for item in history[-8:] if str(item.role).lower().endswith("user")
    ).lower()
    recent_assistant = " ".join(
        item.content for item in history[-8:] if str(item.role).lower().endswith("assistant")
    ).lower()
    asked_for_location = bool(
        re.search(r"\b(?:bus|transport|transportation|route|pickup|pick-up)\b", recent_user)
        and (
            re.search(r"\b(?:my area|my location|cover|come to|serve)\b", recent_user)
            or re.search(r"\b(?:cost|fee|price|charge|how much)\b", recent_user)
            or "few details" in recent_assistant
        )
    )
    return cleaned if asked_for_location else None


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
    if normalized_detail in {
        "pickup_area",
        "pickup_location",
        "location",
        "address",
    }:
        match = re.search(
            r"\b(?:cover|covers|serve|serves|from|at|in|near|around|to|collect|collects|pick up|pickup|pick-up)\s+"
            r"([a-z0-9][a-z0-9 .'-]{1,60}?)(?:[?.!,]|$)",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            return False
        supplied = " ".join(match.group(1).lower().split())
        return supplied not in {
            "me", "my area", "my location", "my address", "our area",
            "our location", "there", "here", "the area", "this area",
        }
    return False


def is_specific_transport_coverage_request(question: str) -> bool:
    """Identify route/locality decisions that require school confirmation."""

    lowered = " ".join(question.lower().split())
    transport = re.search(
        r"\b(?:bus|buses|transport|transportation|route|pickup|pick-up)\b",
        lowered,
    )
    coverage = re.search(
        r"\b(?:come|comes|cover|covers|serve|serves|available|pickup|pick-up|collect|collects|pick)\b",
        lowered,
    )
    personal_place = re.search(
        r"\bmy\s+(?:area|location|address)\b|\bnear\s+me\b",
        lowered,
    ) or detail_is_already_supplied(question, "pickup_area")
    return bool(transport and coverage and personal_place)


def is_broad_transport_coverage_question(question: str) -> bool:
    """Identify questions asking whether every locality is served."""

    lowered = " ".join(question.lower().split())
    return bool(
        re.search(r"\b(?:transport|bus|route)\b", lowered)
        and re.search(r"\b(?:all|every)\s+(?:areas?|localities|neighbou?rhoods?)\b", lowered)
        and re.search(r"\b(?:cover|covered|serve|served|available)\b", lowered)
    )


def deterministic_missing_details(
    question: str,
    context: dict[str, str],
) -> list[str]:
    """Resolve predictable dependent enquiries without an LLM round trip."""

    lowered = f" {' '.join(question.lower().split())} "
    transport_question = bool(
        re.search(
            r"\b(?:bus|buses|transport|transportation|route|pickup|pick-up)\b",
            lowered,
        )
    )
    transport_price_question = transport_question and bool(
        re.search(r"\b(?:cost|costs|fee|fees|price|charge|charges|much)\b", lowered)
    )
    asks_about_personal_location = bool(
        re.search(
            r"\bmy\s+(?:area|location|address)\b|\bnear\s+me\b",
            lowered,
        )
    )
    supplied_location = detail_is_already_supplied(question, "pickup_area") or any(
        context.get(key, "").strip()
        for key in ("pickup_area", "pickup_location", "location", "address")
    )
    if transport_question and (
        asks_about_personal_location or transport_price_question
    ) and not supplied_location:
        return ["pickup_area"]
    return []


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
        top.entry.source_label,
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
        or getattr(top, "evidence_supported", False)
    )


def prioritize_complete_overview(question: str, hits, minimum_score: float):
    """Prefer a complete board overview for broad curriculum questions."""

    if not hits:
        return hits
    terms = typo_tolerant_query_tokens(question)
    if not terms & {"board", "curriculum", "syllabus", "cbse", "icse"}:
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
            approved_by_policy = getattr(hit, "evidence_supported", False)
            dedicated_pathway = f"{requested_pathway} pathway" in title
            pathway_floor = max(0.25, minimum_score * 0.65)
            if requested_pathway not in title or (
                hit.score < minimum_score
                and not approved_by_policy
                and not (dedicated_pathway and hit.score >= pathway_floor)
            ):
                continue
            decision = evaluate_evidence(
                question,
                hit.entry.content,
                hit.score,
                minimum_score,
                hit.entry.source_label,
            )
            content_alignment = expanded_keyword_score(
                question,
                f"{hit.entry.title} {hit.entry.content}",
            )
            if (
                decision.supported
                or approved_by_policy
                or (dedicated_pathway and content_alignment >= 0.35)
            ):
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
            hit.entry.source_label,
        )
        if decision.supported and index:
            return [hit, *hits[:index], *hits[index + 1:]]
        if decision.supported:
            return hits
    return hits


def prioritize_board_guidance(question: str, hits, minimum_score: float):
    """Prefer comparative board guidance when the family asks about fit."""

    if not hits:
        return hits
    lowered = question.lower()
    terms = typo_tolerant_query_tokens(question)
    if not terms & {"board", "cbse", "icse", "curriculum", "syllabus"}:
        return hits
    if not re.search(
        r"\b(?:better|choose|choice|suit|fit|compare|comparison|"
        r"language|languages|strength|strengths|aspiration|aspirations)\b",
        lowered,
    ):
        return hits
    for index, hit in enumerate(hits):
        title = hit.entry.title.lower()
        if hit.score < minimum_score or "choosing between" not in title:
            continue
        decision = evaluate_evidence(
            question,
            hit.entry.content,
            hit.score,
            minimum_score,
            hit.entry.source_label,
        )
        if decision.supported:
            if index:
                return [hit, *hits[:index], *hits[index + 1:]]
            return hits
    return hits


def prioritize_geographic_location(question: str, hits, minimum_score: float):
    """Prefer reviewed location evidence for state/city/address questions."""

    if not hits:
        return hits
    lowered = question.lower()
    terms = typo_tolerant_query_tokens(question)
    # "State board" is an academic-board question, not a geographic one.
    if terms & {"cbse", "icse", "board", "curriculum", "syllabus"}:
        return hits
    asks_for_geography = bool(
        re.search(
            r"\b(?:what|which)\s+(?:state|city|district|country)\b|"
            r"\b(?:state|city|district|country)\s+(?:is|does)\b|"
            r"\bwhere(?:\s+exactly)?\s+(?:is|are)\b|\b(?:located|location|address)\b",
            lowered,
        )
    )
    if not asks_for_geography:
        return hits

    for index, hit in enumerate(hits):
        title = hit.entry.title.lower()
        if hit.score < minimum_score:
            continue
        if "location" not in title and "contact details" not in title:
            continue
        decision = evaluate_evidence(
            question,
            hit.entry.content,
            hit.score,
            minimum_score,
            hit.entry.source_label,
        )
        if decision.supported:
            if index:
                return [hit, *hits[:index], *hits[index + 1:]]
            return hits
    return hits


def prioritize_campus_visit(question: str, hits, minimum_score: float):
    """Prefer visit-booking/contact evidence over incidental campus mentions."""

    if not hits:
        return hits
    lowered = question.lower()
    # A campus visit may be one step inside the admission process. Do not let
    # the visit/contact entry replace the full process entry when the parent is
    # asking how to begin or complete an application.
    if re.search(r"\b(?:admission|application|apply|enquir)\w*\b", lowered) and re.search(
        r"\b(?:start|begin|process|apply|complete)\w*\b", lowered
    ):
        return hits
    if not re.search(
        r"\b(?:visit|visiting|tour|appointment|come|coming|book|booking)\b",
        lowered,
    ):
        return hits
    for index, hit in enumerate(hits):
        title = hit.entry.title.lower()
        if hit.score < minimum_score:
            continue
        if not (
            "visit" in title
            or "contact hours" in title
            or "contact details" in title
        ):
            continue
        decision = evaluate_evidence(
            question,
            hit.entry.content,
            hit.score,
            minimum_score,
            hit.entry.source_label,
        )
        if decision.supported:
            if index:
                return [hit, *hits[:index], *hits[index + 1:]]
            return hits
    return hits


def prioritize_whole_child_learning(question: str, hits, minimum_score: float):
    """Prefer developmental philosophy for questions that move beyond marks."""

    if not hits or not re.search(
        r"\b(?:only marks|beyond marks|what else|whole child|overall development|"
        r"holistic|emotional|social|physical development)\b",
        question,
        flags=re.IGNORECASE,
    ):
        return hits
    for index, hit in enumerate(hits):
        searchable = f"{hit.entry.title} {hit.entry.content}".lower()
        if hit.score < minimum_score:
            continue
        if not re.search(
            r"\b(?:educational philosophy|whole child|intellectual|moral|physical|"
            r"social|emotional|character|leadership)\b",
            searchable,
        ):
            continue
        if index:
            return [hit, *hits[:index], *hits[index + 1:]]
        return hits
    return hits


def prioritize_requested_fact(question: str, hits, minimum_score: float):
    """Prefer the entry that contains the requested entity and fact together."""

    if not hits:
        return hits
    lowered = question.lower()
    requirements: list[tuple[re.Pattern, re.Pattern]] = []
    if re.search(r"\b(?:united states|usa|u\.s\.a\.?|u\.s\.?)\b", lowered):
        requirements.append(
            (
                re.compile(r"location|contact details", re.I),
                re.compile(r"bengaluru|karnataka|rayasandra", re.I),
            )
        )
    if re.search(
        r"\b(?:contact|reach|call|phone|telephone|email|e-mail)\b|"
        r"\b(?:talk|speak)\s+to\s+(?:(?:the\s+)?a\s+)?(?:person|someone|staff|team)\b|"
        r"\b(?:want|what)\s+to\s+(?:talk|speak)\s+to\s+(?:(?:the\s+)?a\s+)?"
        r"(?:person|someone|staff|team)\b",
        lowered,
    ):
        requirements.append(
            (
                re.compile(r"contact details", re.I),
                re.compile(r"phone number.*email address|email address.*phone number", re.I | re.S),
            )
        )
    if re.search(r"\b(?:what|which)\s+days?\b.*\bopen\b|\boffice hours?\b", lowered):
        requirements.append(
            (
                re.compile(r"contact hours|office hours", re.I),
                re.compile(r"monday.*saturday|office hours", re.I | re.S),
            )
        )
    if re.search(
        r"\bwhat\s+time\s+does\s+(?:the\s+)?(?:school|office|campus)\s+open\b|"
        r"\bwhat\s+time\s+(?:can|should)\s+(?:i|we)\s+"
        r"(?:contact|call|phone|reach)\s+(?:the\s+)?school\b|"
        r"\b(?:school|office|campus)\s+(?:opening\s+)?(?:time|timing|timings|hours)\b",
        lowered,
    ):
        requirements.insert(
            0,
            (
                re.compile(r"contact hours|office hours", re.I),
                re.compile(r"office hours.*monday.*saturday", re.I | re.S),
            ),
        )
    if re.search(r"\b(?:where|located|location|address|state|city)\b", lowered):
        requirements.append((re.compile(r"location|contact details", re.I), re.compile(r"rayasandra|bengaluru|karnataka|located", re.I)))
    if "icse" in lowered and re.search(r"\b(?:subject|include|learning area)\w*\b", lowered):
        requirements.append((re.compile(r"icse pathway", re.I), re.compile(r"english.*mathematics|computer applications|learning areas", re.I | re.S)))
    if "cbse" in lowered and re.search(r"\b(?:subject|include|syllabus|pathway)\w*\b", lowered):
        requirements.append((re.compile(r"cbse pathway", re.I), re.compile(r"ncert|mathematics|science|cbse syllabus", re.I)))
    if re.search(r"\b(?:book|booking|appointment)\b", lowered) and re.search(r"\b(?:visit|campus)\b", lowered):
        requirements.append((re.compile(r"visit|contact hours", re.I), re.compile(r"booked in advance|book in advance|appointment", re.I)))
    if re.search(r"\b(?:arrange|schedule|book|make)\w*\b", lowered) and re.search(
        r"\b(?:enquiry|calling admissions|how can i)\b", lowered
    ):
        requirements.insert(
            0,
            (
                re.compile(r"admission enquiry information", re.I),
                re.compile(r"call.*admissions.*arrange a visit|counsellor.*arrange a visit", re.I | re.S),
            ),
        )
    if re.search(r"\b(?:visit|tour|come|see)\b", lowered) and "campus" in lowered:
        requirements.append((re.compile(r"visit|contact hours", re.I), re.compile(r"booked in advance|campus visits|office hours", re.I)))
    if re.search(r"\b(?:assess|assessment|exam|examination|rank)\w*\b", lowered):
        requirements.append((re.compile(r"learning and assessment", re.I), re.compile(r"assessment|rank order|projects|unit tests", re.I)))
    if re.search(r"\b(?:meal|food).*(?:partner|provider)|(?:partner|provider).*(?:meal|food)\b", lowered):
        requirements.append((re.compile(r"meal", re.I), re.compile(r"sodexo|meal-service partner", re.I)))
    if re.search(r"\b(?:allerg|dietary|intolerance)\w*\b", lowered):
        requirements.append((re.compile(r"meal", re.I), re.compile(r"allerg|dietary", re.I)))
    if re.search(r"\bgrade\s*12\b|\bextend\b|\bgrow\b", lowered):
        requirements.append((re.compile(r"grades and curriculum", re.I), re.compile(r"grade 12|reaching grade 12", re.I)))
    if re.search(r"\b(?:admission|application|apply|enquir)\w*\b", lowered) and re.search(
        r"\b(?:start|begin|process|apply|complete)\w*\b", lowered
    ):
        requirements.append(
            (
                re.compile(r"admission process", re.I),
                re.compile(r"enquir.*visit the campus.*application", re.I | re.S),
            )
        )
    if not requirements:
        return hits

    # Exact title-and-content agreement is stronger than the aggregate hybrid
    # score for short questions, so allow a low floor without accepting a
    # merely related passage.
    floor = max(0.10, minimum_score * 0.30)
    for title_pattern, content_pattern in requirements:
        for index, hit in enumerate(hits):
            if hit.score < floor:
                continue
            if title_pattern.search(hit.entry.title) and content_pattern.search(hit.entry.content):
                if index:
                    return [hit, *hits[:index], *hits[index + 1:]]
                return hits
    return hits


def is_clear_unsupported_school_requirement(question: str) -> bool:
    """Skip misleading suggestions for requirements needing staff confirmation."""

    return bool(re.search(
        r"\b(?:wheelchair|accessible|accessibility|mobility|medical|medication|"
        r"special accommodation|reasonable accommodation|individual support need)\b",
        question,
        flags=re.IGNORECASE,
    ))


def _clean_source_attribution(sentence: str) -> str:
    """Remove source-reporting boilerplate without changing the underlying fact."""

    replacements = (
        (r"^The Contact page publishes office hours as\s+", "Office hours are "),
        (r"^The school website states that\s+", ""),
        (r"^The school states that\s+", ""),
        (r"^The official website states that\s+", ""),
        (r"^The official website says that\s+", ""),
        (r"^The official website describes it as\s+", "It is "),
        (r"^The brochure says that\s+", ""),
        (r"^The official brochure states that\s+", ""),
        (r"^It says the\s+", "The "),
        (r"^The school says it does not\s+", "The school does not "),
    )
    cleaned = sentence.strip()
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if updated != cleaned:
            cleaned = updated
            break
    if cleaned:
        cleaned = re.sub(
            r",\s+and says email replies are normally made within one working day",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r",\s+along with extended day care for families needing a longer day",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _split_source_sentences(content: str) -> list[str]:
    """Split prose without breaking names after common honorifics."""

    marker = "<HONORIFIC_PERIOD>"
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Miss|Dr|Prof)\.",
        lambda match: f"{match.group(1)}{marker}",
        content.strip(),
        flags=re.IGNORECASE,
    )
    return [
        sentence.replace(marker, ".").strip()
        for sentence in re.split(r"(?<=[.!?])\s+", protected)
        if sentence.strip()
    ]


def concise_source_answer(
    response_question: str,
    evidence_question: str,
    content: str,
) -> str:
    """Present approved evidence as a direct, short conversational answer."""

    sentences = _split_source_sentences(content)
    if not sentences:
        return content.strip()

    response_lower = response_question.lower()
    evidence_lower = evidence_question.lower()

    if re.search(r"\b(?:united states|usa|u\.s\.a\.?|u\.s\.?)\b", response_lower) and re.search(
        r"\b(?:bengaluru|karnataka|rayasandra)\b",
        content,
        re.I,
    ):
        location = next(
            (
                _clean_source_attribution(sentence)
                for sentence in sentences
                if re.search(r"\b(?:located|address)\b", sentence, re.I)
                and re.search(r"\b(?:bengaluru|karnataka|rayasandra)\b", sentence, re.I)
            ),
            None,
        )
        if location:
            return f"No. {location}"

    asks_contact_time = bool(re.search(
        r"\bwhat\s+time\s+(?:can|should)\s+(?:i|we)\s+"
        r"(?:contact|call|phone|reach)\s+(?:the\s+)?school\b",
        response_lower,
    ))
    asks_phone = bool(re.search(r"\b(?:phone|telephone|mobile|call)\b", response_lower))
    asks_email = bool(re.search(r"\b(?:email|e-mail)\b", response_lower))
    asks_contact = bool(
        re.search(r"\b(?:contact|reach)\b", response_lower)
        or re.search(
            r"\b(?:talk|speak)\s+to\s+(?:(?:the\s+)?a\s+)?(?:person|someone|staff|team)\b|"
            r"\b(?:want|what)\s+to\s+(?:talk|speak)\s+to\s+(?:(?:the\s+)?a\s+)?"
            r"(?:person|someone|staff|team)\b",
            response_lower,
        )
    )
    if (asks_phone or asks_email or asks_contact) and not asks_contact_time:
        contact_sentences = [
            _clean_source_attribution(sentence)
            for sentence in sentences
            if (
                (asks_phone and re.search(r"\b(?:phone|telephone|mobile|call)\b|\+\d", sentence, re.I))
                or (asks_email and re.search(r"\bemail\b|[\w.+-]+@[\w.-]+", sentence, re.I))
                or (
                    asks_contact
                    and re.search(r"\b(?:phone|telephone|email)\b|\+\d|[\w.+-]+@[\w.-]+", sentence, re.I)
                )
            )
        ]
        if contact_sentences:
            return " ".join(dict.fromkeys(contact_sentences[:2]))

    if re.search(r"\b(?:what|which)\s+days?\b.*\bopen\b", response_lower):
        office_days = next(
            (
                _clean_source_attribution(sentence)
                for sentence in sentences
                if re.search(r"\bmonday\b.*\bsaturday\b", sentence, re.I)
            ),
            None,
        )
        if office_days:
            return office_days

    if re.search(
        r"\bwhat\s+time\s+does\s+(?:the\s+)?(?:school|office|campus)\s+open\b|"
        r"\bwhat\s+time\s+(?:can|should)\s+(?:i|we)\s+"
        r"(?:contact|call|phone|reach)\s+(?:the\s+)?school\b|"
        r"\b(?:school|office|campus)\s+(?:opening\s+)?(?:time|timing|timings|hours)\b",
        f"{response_lower} {evidence_lower}",
    ):
        office_hours = next(
            (
                _clean_source_attribution(sentence)
                for sentence in sentences
                if re.search(r"\boffice hours\b.*\bmonday\b.*\bsaturday\b", sentence, re.I)
            ),
            None,
        )
        if office_hours:
            return office_hours

    benefit_request = bool(re.search(
        r"\b(?:how\s+(?:does|do)\s+.*\bhelp|what\s+(?:abilities|skills|benefits)|"
        r"benefits?\s+(?:does|do)|explain.*benefits?|benefits?.*simply)\b",
        response_lower,
    ))

    if re.search(r"\bgrade\s*12\b", response_lower):
        grade_12 = next(
            (sentence for sentence in sentences if re.search(r"\bgrade\s*12\b", sentence, re.I)),
            None,
        )
        if grade_12:
            fact = _clean_source_attribution(grade_12)
            return f"Yes. {fact}" if re.match(r"\s*(?:will|does|is|can)\b", response_question, re.I) else fact

    if re.search(r"\b(?:explain|describe|details?|more)\b", response_lower):
        if "icse" in response_lower and re.search(r"\blanguage-rich\b", content, re.I):
            return " ".join(_clean_source_attribution(sentence) for sentence in sentences[:2])
        if "cbse" in response_lower and re.search(r"\bncert\b", content, re.I):
            return " ".join(_clean_source_attribution(sentence) for sentence in sentences[:2])

    # Recommendations are framed as evidence-based guidance, not certainty.
    if re.search(r"\b(?:which|what)\s+(?:one|board).*\b(?:suit|fit|better)\b|\bmay suit\b", response_lower):
        combined = f"{evidence_lower} {response_lower}"
        if re.search(r"\b(?:language|languages|writing|literature)\b", combined) and re.search(
            r"\bICSE\b[^.]*\b(?:language-rich|language|analytically focused)\b",
            content,
            flags=re.IGNORECASE,
        ):
            return (
                "ICSE is described as broad, language-rich and analytically focused, "
                "so it may be a suitable option."
            )
        if re.search(r"\b(?:science|mathematics|maths|portable|move|moving|cities|jee|neet|cuet)\b", combined) and re.search(
            r"\bCBSE\b[^.]*\b(?:NCERT|portable|Mathematics|Science|JEE|NEET|CUET)\b",
            content,
            flags=re.IGNORECASE,
        ):
            return (
                "CBSE is described as NCERT-aligned, nationally portable, and strong "
                "in Mathematics and Science foundations, so it may be a suitable option."
            )

    # Polarity-sensitive questions must use an explicit approved positive or
    # negative statement. Never infer "Yes" merely because the topic matched.
    if re.search(r"\b(?:rank|ranking|ranked)\b", response_lower):
        no_rank = re.search(
            r"(?:The school says it )?does not publish a rank order[^.]*\.?",
            content,
            flags=re.IGNORECASE,
        )
        if no_rank:
            fact = _clean_source_attribution(no_rank.group(0).strip())
            return f"No. {fact}"

    if re.search(r"\b(?:only|solely)\b.*\b(?:exam|exams|examination|examinations|marks|tests)\b", response_lower):
        varied = [
            _clean_source_attribution(sentence) for sentence in sentences
            if re.search(r"\b(?:observation|discussion|journals?|peer assessment|self-assessment|projects?|presentations?|unit tests?)\b", sentence, re.I)
        ]
        if varied:
            return f"No. {varied[0]}"

    # General assessment questions should foreground the actual assessment
    # methods, not an adjacent sentence about learning styles. Keep the answer
    # entirely extractive by selecting the approved assessment sentences.
    if re.search(r"\b(?:assess|assessment|evaluate|evaluation)\w*\b", response_lower):
        assessment_sentences = [
            _clean_source_attribution(sentence)
            for sentence in sentences
            if re.search(
                r"\b(?:assessment|assessed|observation|discussion|journals?|"
                r"peer assessment|self-assessment|projects?|presentations?|"
                r"unit tests?|learning portfolio)\b",
                sentence,
                flags=re.IGNORECASE,
            )
        ]
        if assessment_sentences:
            return " ".join(assessment_sentences[:2])

    # Provider, dietary-requirement and activity-list questions should return
    # the matching approved fact instead of the first sentence in the entry.
    if re.search(r"\b(?:meal|food)[-\s]+(?:service[-\s]+)?(?:partner|provider)\b|\bwho\s+(?:provides|supplies|serves)\b", response_lower):
        provider = next(
            (
                sentence for sentence in sentences
                if re.search(r"\b(?:provider|partner|sodexo|provided by)\b", sentence, re.I)
            ),
            None,
        )
        if provider:
            return _clean_source_attribution(provider)

    if re.search(r"\b(?:allerg|dietary|diet restriction|intolerance)\w*\b", response_lower):
        dietary = next(
            (
                sentence for sentence in sentences
                if re.search(r"\b(?:allerg|dietary|intolerance|special diet)\w*\b", sentence, re.I)
            ),
            None,
        )
        if dietary:
            return _clean_source_attribution(dietary)

    if re.search(
        r"\b(?:is|are)\s+(?:food|meals?)\s+(?:served|available)\b|"
        r"\b(?:does|do)\s+.*\b(?:provide|serve|offer)\s+(?:food|meals?)\b",
        response_lower,
    ):
        availability = next(
            (
                sentence for sentence in sentences
                if re.search(r"\b(?:meals?|food)\b", sentence, re.I)
                and re.search(r"\b(?:served|provide|prepared|cooked|available|designed)\w*\b", sentence, re.I)
                and "optional" not in sentence.lower()
            ),
            None,
        )
        if availability:
            return f"Yes. {_clean_source_attribution(availability)}"

    if re.search(r"\b(?:compulsory|mandatory|required|optional)\b", response_lower) and re.search(
        r"\b(?:meal|meals|food)\b", response_lower
    ):
        optional = next(
            (sentence for sentence in sentences if re.search(r"\bmeals?\b[^.]*\boptional\b", sentence, re.I)),
            None,
        )
        if optional:
            fact = _clean_source_attribution(optional)
            return f"No. {fact}" if re.search(r"\b(?:compulsory|mandatory|required)\b", response_lower) else fact

    if re.search(r"\b(?:book|booking|appointment|reserve)\b", response_lower) and re.search(
        r"\b(?:visit|come|coming|campus|before)\b", response_lower
    ):
        booking = next(
            (sentence for sentence in sentences if re.search(r"\b(?:booked in advance|book in advance|appointment)\b", sentence, re.I)),
            None,
        )
        if booking:
            return _clean_source_attribution(booking)

    if re.search(r"\bhow\b.*\b(?:arrange|schedule|book|make)\w*\b", evidence_lower) and re.search(
        r"\b(?:visit|campus|appointment)\b", evidence_lower
    ):
        direct_method_sentences = [
            _clean_source_attribution(sentence)
            for sentence in sentences
            if re.search(
                r"\b(?:call|phone|counsellor|arrange a visit|visit slot)\w*\b",
                sentence,
                re.I,
            )
        ]
        method_sentences = direct_method_sentences or [
            _clean_source_attribution(sentence)
            for sentence in sentences
            if re.search(r"\b(?:enquir|submit)\w*\b", sentence, re.I)
        ]
        if method_sentences:
            return " ".join(method_sentences[:2])

    asks_for_activity_list = bool(
        re.search(
            r"\b(?:what|which)\s+(?:sports?|physical activities|activities)\b|"
            r"\b(?:sports?|physical activities).*(?:available|offer|participate)\b",
            response_lower,
        )
    )
    if asks_for_activity_list:
        activity_sentence = next(
            (
                sentence for sentence in sentences
                if len(re.findall(
                    r"\b(?:basketball|volleyball|table tennis|badminton|swimming|"
                    r"cricket|futsal|indoor games)\b",
                    sentence,
                    flags=re.IGNORECASE,
                )) >= 2
            ),
            None,
        )
        if activity_sentence:
            return _clean_source_attribution(activity_sentence)

    # Current fee questions need the availability statement first. A category
    # list alone must not look like the requested exact amount.
    if not re.search(r"\bwhy\b", response_lower) and re.search(
        r"\b(?:exact|current|this year)\b.*\b(?:fee|fees|cost|price)\b|"
        r"\bfee\b.*\b(?:exact|current|this year)\b",
        response_lower,
    ):
        unavailable = next(
            (
                sentence for sentence in sentences
                if re.search(r"\b(?:does not publish exact fee|exact amounts? must be confirmed|request the current figures)\b", sentence, re.I)
            ),
            None,
        )
        if unavailable:
            return _clean_source_attribution(unavailable)

    # A process-stage follow-up should continue from the named stage instead of
    # repeating the complete approved process. The returned clauses remain
    # verbatim source facts, with only capitalization adjusted.
    if re.search(
        r"\b(?:after|following)\s+(?:the\s+)?(?:campus\s+)?visit(?:ing)?\b|"
        r"\bwhat comes after\s+visit(?:ing)?\b|"
        r"\bafter\s+(?:submitting|completing)\s+(?:the\s+)?(?:application|form)\b|"
        r"\bstep\s+after\s+(?:submitting|completing)\s+(?:the\s+)?(?:application|form)\b",
        response_lower,
    ):
        clauses = [part.strip(" .") for part in re.split(r"[;]", content) if part.strip()]
        stage_pattern = (
            r"visit the campus" if "visit" in response_lower
            else r"complete the application|submitting the application"
        )
        visit_index = next(
            (index for index, clause in enumerate(clauses) if re.search(stage_pattern, clause, re.I)),
            None,
        )
        if visit_index is not None and visit_index + 1 < len(clauses):
            following = clauses[visit_index + 1:visit_index + 3]
            focused = "; then ".join(
                re.sub(r"^(?:then\s+)", "", clause, flags=re.IGNORECASE)
                for clause in following
            )
            focused = focused[0].upper() + focused[1:]
            return focused.rstrip(".") + "."

    # Subject-list questions require the approved curriculum enumeration, not
    # the preceding generic sentence identifying the board.
    if re.search(
        r"\b(?:what|which)\s+(?:subjects?|learning areas?)\b|"
        r"\bsubjects?\s+does\s+.*\binclude\b",
        response_lower,
    ):
        subject_terms = re.compile(
            r"\b(?:english|language|mathematics|environmental science|science|"
            r"social studies|computer applications|value education|art|craft|"
            r"physical education)\b",
            flags=re.IGNORECASE,
        )
        subject_sentences = [
            sentence for sentence in sentences
            if len(subject_terms.findall(sentence)) >= 3
        ]
        if subject_sentences:
            subject_sentence = subject_sentences[0]
            listed = re.search(
                r"\bwith\s+(.+?)\s+among the learning areas(?:\s+for\s+[^.]*)?\.?$",
                subject_sentence,
                flags=re.IGNORECASE,
            )
            if listed:
                return f"The curriculum includes {listed.group(1).strip().rstrip('.')}."
            return _clean_source_attribution(subject_sentence)

    # Return only the geographic entity requested, provided that the entity is
    # explicitly present in the approved passage.
    if re.search(r"\b(?:what|which)\s+state\b", response_lower):
        indian_states = (
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
            "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
            "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
            "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
            "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
            "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
            "West Bengal", "Delhi", "Chandigarh", "Jammu and Kashmir",
            "Ladakh", "Puducherry",
        )
        for state in indian_states:
            if re.search(rf"\b{re.escape(state)}\b", content, flags=re.IGNORECASE):
                return f"{state}."

    # A useful "why" answer needs both the approved policy/condition and its
    # approved explanation. Do not return the explanation without identifying
    # what it explains.
    if re.search(r"\bwhy\b", response_lower):
        policy_sentences = [
            sentence for sentence in sentences
            if re.search(
                r"\b(?:does not|do not|is not|are not|not published|not available)\b",
                sentence,
                flags=re.IGNORECASE,
            )
            and expanded_keyword_score(response_question, sentence) > 0
        ]
        explanation_sentences = [
            sentence for sentence in sentences
            if re.search(
                r"\b(?:because|depends|vary|varies|shared directly|current structure|"
                r"grade-specific|optional|route|distance)\b",
                sentence,
                flags=re.IGNORECASE,
            )
        ]
        if policy_sentences and explanation_sentences:
            combined = [policy_sentences[0]]
            if explanation_sentences[0] != policy_sentences[0]:
                combined.append(explanation_sentences[0])
            return " ".join(_clean_source_attribution(sentence) for sentence in combined)

    scored = []
    for index, sentence in enumerate(sentences):
        score = max(
            expanded_keyword_score(response_question, sentence),
            expanded_keyword_score(evidence_question, sentence),
        )
        sentence_lower = sentence.lower()
        if re.search(r"\bwho\s+(?:provides|supplies|serves)\b", response_lower) and re.search(
            r"\b(?:provider|provided by|partner|sodexo|supplies|serves)\b",
            sentence_lower,
        ):
            score += 1.0
        if re.search(r"\b(?:what time|when|hours?)\b", response_lower) and re.search(
            r"\b(?:hours?|am|pm|monday|saturday|opening|closing)\b",
            sentence_lower,
        ):
            score += 1.0
        if benefit_request and re.search(
            r"\b(?:help|develop|skills?|practise|practice|language|reasoning|problem-solving)\b",
            sentence_lower,
        ):
            score += 2.0
        if re.search(r"\bwhat\s+sports?\b|\bsports?.*(?:participate|available|offer)\b", response_lower):
            if re.search(
                r"\b(?:basketball|volleyball|table tennis|badminton|swimming|"
                r"cricket|futsal|indoor games)\b",
                sentence_lower,
            ):
                score += 2.0
        if re.search(r"\b(?:include|includes|including)\s+swimming\b", response_lower):
            if "swimming" in sentence_lower:
                score += 2.0
        if re.search(r"\bafter\s+(?:the\s+)?campus visit\b", response_lower):
            if re.search(r"\b(?:complete the application|documents|interaction|offer|onboarding)\b", sentence_lower):
                score += 2.0
        if re.search(r"\b(?:language|languages)\b", response_lower) and re.search(
            r"\b(?:language-rich|languages?|icse)\b",
            sentence_lower,
        ):
            score += 1.0
        if re.search(r"\bwhy\b", response_lower) and re.search(
            r"\b(?:because|depends|vary|varies|shared directly|current structure|"
            r"grade-specific|optional|route|distance)\b",
            sentence_lower,
        ):
            score += 1.5
        scored.append((score, index, sentence))
    best_score = max(score for score, _, _ in scored)
    selected = [
        (index, sentence)
        for score, index, sentence in scored
        if score > 0 and score >= best_score * 0.45
    ]
    if not selected:
        selected = [(0, sentences[0])]
    selected = sorted(selected)[:2]
    answer = " ".join(_clean_source_attribution(sentence) for _, sentence in selected)

    if re.search(
        r"\b(?:(?:tell|explain)\s+me(?:\s+more)?|could you explain)\b",
        response_lower,
    ):
        informative = [
            _clean_source_attribution(sentence)
            for sentence in sentences[:2]
        ]
        answer = " ".join(informative)

    if benefit_request:
        benefit_sentences = [
            _clean_source_attribution(sentence)
            for sentence in sentences
            if re.search(
                r"\b(?:practise|practice|develop|skills?|language|motor|cooperation|"
                r"reasoning|problem-solving|scientific thinking)\b",
                sentence,
                flags=re.IGNORECASE,
            )
        ]
        if benefit_sentences:
            answer = " ".join(benefit_sentences[:1])

    if re.search(r"\b(?:include|includes|including)\s+swimming\b|\bswimming\s+(?:one|among)\b", response_lower):
        if "swimming" in content.lower():
            answer = "Swimming is listed among the school’s sports facilities."

    if re.search(r"\b(?:language|languages)\b", response_lower):
        comparison = re.search(
            r"(?:while\s+)?(ICSE\s+is\s+described\s+as\s+[^.]*language-rich[^.]*)",
            answer,
            flags=re.IGNORECASE,
        )
        if comparison:
            focused = comparison.group(1).strip(" ,")
            answer = focused[0].upper() + focused[1:] + "."

    direct_question = correct_query_typos(re.sub(
        r"^\s*(?:(?:hello|hi|thanks|thank you|okay|ok|great)[!. ,]*)+",
        "",
        response_question,
        flags=re.IGNORECASE,
    ))
    asks_yes_no = bool(
        re.match(
            r"\s*(?:does|do|did|is|are|was|were|can|could|will|would|has|have|may)\b",
            direct_question,
            flags=re.IGNORECASE,
        )
    )
    if re.match(r"\s*(?:can|could|would)\s+you\s+(?:tell|explain|describe|show)\b", direct_question, re.I):
        asks_yes_no = False
    if asks_yes_no and not re.match(r"\s*(?:yes|no)\b", answer, flags=re.IGNORECASE):
        negative = bool(
            re.search(
                r"\b(?:does not|do not|is not|are not|no entrance test|not available|"
                r"doesn't|don't|isn't|aren't|not compulsory|not mandatory)\b",
                answer,
                flags=re.IGNORECASE,
            )
        )
        answer = f"{'No' if negative else 'Yes'}. {answer}"
    return answer


def extractive_fast_answer(
    question: str,
    hits,
    minimum_score: float,
    response_question: str | None = None,
) -> str | None:
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
        top.entry.source_label,
    )
    if not decision.supported:
        return None

    # Referential benefit questions can have little lexical overlap on their
    # own ("How does that help them?"). When the bounded retrieval question and
    # approved passage clearly contain the benefit, answer extractively instead
    # of invoking generation.
    current_question = response_question or question
    if re.search(
        r"\b(?:how\s+(?:does|do)\s+.*\bhelp|what\s+(?:abilities|skills|benefits)|"
        r"explain.*benefits?|benefits?.*simply)\b",
        current_question,
        flags=re.IGNORECASE,
    ) and re.search(
        r"\b(?:practise|practice|develop|skills?|language|motor|cooperation|"
        r"reasoning|problem-solving|scientific thinking)\b",
        content,
        flags=re.IGNORECASE,
    ):
        return concise_source_answer(current_question, question, content)

    # Exact fact selectors are safe to answer without generation when the
    # selected published passage explicitly contains the requested fact.
    focused_fact = bool(
        (re.search(r"\bsubjects?\b", current_question, re.I) and re.search(r"\b(?:english|mathematics|computer applications|learning areas)\b", content, re.I))
        or (re.search(r"\b(?:book|booking)\b", current_question, re.I) and re.search(r"\bbooked in advance\b", content, re.I))
        or (re.search(r"\bwhere\b", current_question, re.I) and re.search(r"\b(?:rayasandra|bengaluru|karnataka)\b", content, re.I))
        or (re.search(r"\b(?:suit|fit|better)\b", current_question, re.I) and re.search(r"\b(?:language-rich|nationally portable|mathematics and science)\b", content, re.I))
        or (re.search(r"\b(?:jee|neet|cuet)\b", current_question, re.I) and re.search(r"\b(?:jee|neet|cuet)\b", content, re.I))
    )
    if focused_fact:
        return concise_source_answer(current_question, question, content)
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
        top.score < 0.50
        and lexical_coverage < 0.60
        and synonym_match < 1.0
        and not deterministic_typo_match
        and not getattr(top, "evidence_supported", False)
    ):
        title_coverage = meaningful_keyword_score(question, top.entry.title)
        contextual_followup = bool(
            response_question
            and response_question.strip() != question.strip()
        )
        contextual_title_match = contextual_followup and title_coverage >= 0.25
        required_score = minimum_score if contextual_title_match else 0.42
        if (
            top.score < required_score
            or (title_coverage < 0.50 and not contextual_title_match)
        ):
            return None
    if not content or len(content) > 700:
        return None
    return concise_source_answer(
        response_question or question,
        question,
        content,
    )


def grade_specific_age_is_missing(question: str, hits) -> bool:
    """Prevent a broad age band from masquerading as a grade-specific cutoff."""

    grade = re.search(r"\b(?:grade|class)\s*(\d{1,2})\b", question, re.I)
    asks_age = re.search(r"\b(?:age|old|birth|dob|eligib)\w*\b", question, re.I)
    if not grade or not asks_age or not hits:
        return False
    grade_number = grade.group(1)
    grade_age_fact = re.compile(
        rf"\b(?:grade|class)\s*{re.escape(grade_number)}\b[^.]*"
        r"\b(?:age|years? old|born|date of birth|dob)\b",
        re.I,
    )
    return not any(grade_age_fact.search(hit.entry.content) for hit in hits)

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

        if is_prompt_injection(request.message):
            return ChatResponse(
                conversation_id=conversation_id,
                answer=(
                    "I can only answer using approved school information, so I can’t "
                    "invent or guess details."
                ),
                outcome="out_of_scope",
                confidence=1.0,
            )

        if contains_unsupported_negative_premise(request.message):
            return ChatResponse(
                conversation_id=conversation_id,
                answer=(
                    "I don’t have approved information indicating that statement is "
                    "true. Please contact the school if you would like confirmation "
                    "about the relevant admission eligibility."
                ),
                outcome="answered",
                confidence=0.0,
            )

        friendly_answer = social_response(request.message)
        if friendly_answer:
            return ChatResponse(
                conversation_id=conversation_id,
                answer=friendly_answer,
                outcome="answered",
                confidence=1.0,
            )

        acknowledgement = context_acknowledgement(request.message)
        if acknowledgement:
            return ChatResponse(
                conversation_id=conversation_id,
                answer=acknowledgement,
                outcome="answered",
                confidence=1.0,
            )

        continued_pickup_area = transport_location_continuation(
            request.message,
            request.history,
        )
        if not continued_pickup_area and supplied_pickup_area(request.message):
            recent_user_context = correct_query_typos(" ".join(
                item.content
                for item in request.history[-8:]
                if str(item.role).lower().endswith("user")
            ))
            if re.search(
                r"\b(?:bus|transport|route|pickup|pick-up)\b",
                recent_user_context,
                flags=re.IGNORECASE,
            ) and re.search(
                r"\b(?:pick up|pickup|pick-up|pick|cover|serve|come)\b",
                correct_query_typos(request.message),
                flags=re.IGNORECASE,
            ):
                continued_pickup_area = supplied_pickup_area(request.message)
        if continued_pickup_area:
            contextual_ticket = escalation_question(
                "Does the school transport route cover this pickup area?",
                {**request.context, "pickup_area": continued_pickup_area},
                request.history,
            )
            unanswered = self.unanswered.create(
                request.school_id,
                contextual_ticket,
            )
            return ChatResponse(
                conversation_id=conversation_id,
                answer=(
                    "I don’t have approved route coverage information for "
                    f"{continued_pickup_area}. I’ve sent the location and your "
                    "question to the school team for confirmation. You may add "
                    "your email below if you would like to receive their answer "
                    "directly."
                ),
                outcome="escalated",
                confidence=0.0,
                unanswered_id=unanswered.id,
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

        # Broad coverage is live, route-dependent information. Resolve it with
        # conversation history before the generic dynamic-information gate so
        # the family receives a useful, contextual escalation rather than the
        # unrelated fallback message.
        early_retrieval_context = resolve_contextual_question(
            contextual_question,
            request.history,
        )
        if is_broad_transport_coverage_question(early_retrieval_context):
            unanswered = self.unanswered.create(request.school_id, ticket_question)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=(
                    "Not necessarily. Transport coverage depends on route feasibility, "
                    "timings and availability for the family’s location. I’ve sent your "
                    "question to the school team so they can confirm the applicable route."
                ),
                outcome="escalated",
                confidence=0.0,
                unanswered_id=unanswered.id,
            )

        if (
            is_dynamic_operational_question(scope_question)
            and not is_named_staff_question(scope_question)
        ):
            unanswered = self.unanswered.create(request.school_id, ticket_question)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=ESCALATION_MESSAGE,
                outcome="escalated",
                confidence=0.0,
                unanswered_id=unanswered.id,
            )

        # Try approved hybrid retrieval first. Clear, general questions do not need a
        # separate LLM classification call; this removes one full local-model round trip.
        retrieval_context = resolve_contextual_question(
            contextual_question,
            request.history,
        )
        requested_staff_grades: set[str] = set()
        if is_named_staff_question(retrieval_context):
            requested_staff_grades = grades_from_context(request.context)
            if not requested_staff_grades:
                requested_staff_grades = mentioned_grades(request.message)
            if not requested_staff_grades:
                requested_staff_grades = most_recent_user_grades(request.history)
            if not requested_staff_grades:
                return ChatResponse(
                    conversation_id=conversation_id,
                    answer=(
                        "Which grade or class are you asking about? Please provide "
                        "the grade so I can check the correct teacher information."
                    ),
                    outcome="clarification",
                    confidence=0.0,
                    clarification_question=request.message,
                    clarification_kind="details",
                    required_details=["grade"],
                )
            # Canonicalize the staff query so stale grade mentions from older
            # turns cannot influence retrieval or central evidence validation.
            grade_text = " and ".join(
                f"Grade {grade}" for grade in sorted(requested_staff_grades)
            )
            retrieval_context = f"Who is the teacher for {grade_text}?"
        hits = self.knowledge.search(request.school_id, retrieval_context)

        # Staff assignments change and must never be inferred from general
        # website or brochure text. A currently active response explicitly
        # published by the school authority is the only permitted exception.
        if is_named_staff_question(retrieval_context):
            grade_question = " ".join(
                f"Grade {grade}" for grade in sorted(requested_staff_grades)
            )
            authority_hits = [
                hit
                for hit in hits
                if is_school_authority_response(hit)
                and (hit.score >= self.min_score or hit.evidence_supported)
                and staff_hit_matches_requested_grade(grade_question, hit)
            ]
            if not authority_hits:
                unanswered = self.unanswered.create(request.school_id, ticket_question)
                return ChatResponse(
                    conversation_id=conversation_id,
                    answer=ESCALATION_MESSAGE,
                    outcome="escalated",
                    confidence=hits[0].score if hits else 0.0,
                    unanswered_id=unanswered.id,
                )
            hits = authority_hits
        hits = prioritize_complete_overview(
            retrieval_context,
            hits,
            self.min_score,
        )
        hits = prioritize_board_guidance(
            retrieval_context,
            hits,
            self.min_score,
        )
        hits = prioritize_geographic_location(
            retrieval_context,
            hits,
            self.min_score,
        )
        hits = prioritize_campus_visit(
            retrieval_context,
            hits,
            self.min_score,
        )
        hits = prioritize_whole_child_learning(
            retrieval_context,
            hits,
            self.min_score,
        )
        hits = prioritize_requested_fact(
            retrieval_context,
            hits,
            self.min_score,
        )
        predictable_missing_details = deterministic_missing_details(
            retrieval_context,
            request.context,
        )
        if predictable_missing_details:
            return ChatResponse(
                conversation_id=conversation_id,
                answer=DETAILS_CLARIFICATION_MESSAGE,
                outcome="clarification",
                confidence=0.0,
                clarification_question=request.message,
                clarification_kind="details",
                required_details=predictable_missing_details,
            )
        if is_broad_transport_coverage_question(retrieval_context):
            unanswered = self.unanswered.create(request.school_id, ticket_question)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=(
                    "Not necessarily. Transport coverage depends on route feasibility, "
                    "timings and availability for the family’s location. I’ve sent your "
                    "question to the school team so they can confirm the applicable route."
                ),
                outcome="escalated",
                confidence=hits[0].score if hits else 0.0,
                unanswered_id=unanswered.id,
            )
        if (
            is_specific_transport_coverage_request(request.message)
            and detail_is_already_supplied(request.message, "pickup_area")
        ):
            pickup_area = supplied_pickup_area(request.message)
            unanswered = self.unanswered.create(request.school_id, ticket_question)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=(
                    "I don’t have approved route coverage information for "
                    f"{pickup_area}. I’ve sent the location and your question to "
                    "the school team for confirmation. You may add your email "
                    "below if you would like to receive their answer directly."
                    if pickup_area
                    else ESCALATION_MESSAGE
                ),
                outcome="escalated",
                confidence=0.0,
                unanswered_id=unanswered.id,
            )
        deterministic_typo_match = (
            has_likely_typo(retrieval_context)
            and has_strong_approved_hit(
                retrieval_context,
                hits,
                self.min_score,
            )
        )
        if deterministic_typo_match or is_clear_missing_information(
            contextual_question
        ) or (
            not needs_contextual_understanding(retrieval_context) and hits
        ):
            understanding = QueryUnderstanding(
                normalized_question=retrieval_context,
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

        retrieval_parts = [retrieval_context if followup else understanding.normalized_question]
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
            hits = prioritize_board_guidance(
                retrieval_question,
                hits,
                self.min_score,
            )
            hits = prioritize_geographic_location(
                retrieval_question,
                hits,
                self.min_score,
            )
            hits = prioritize_campus_visit(
                retrieval_question,
                hits,
                self.min_score,
            )
            hits = prioritize_whole_child_learning(
                retrieval_question,
                hits,
                self.min_score,
            )
            hits = prioritize_requested_fact(
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
            if (
                candidates
                and not is_clear_missing_information(retrieval_question)
                and not is_clear_unsupported_school_requirement(retrieval_question)
            ):
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
            escalation_answer = (
                "Not necessarily. Transport coverage depends on route feasibility, "
                "timings and availability for the family’s location. I’ve sent your "
                "question to the school team so they can confirm the applicable route."
                if (
                    is_broad_transport_coverage_question(retrieval_context)
                    or is_broad_transport_coverage_question(retrieval_question)
                )
                else ESCALATION_MESSAGE
            )
            return ChatResponse(
                conversation_id=conversation_id,
                answer=escalation_answer,
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
        # Use the deterministic standalone form for follow-ups. Model-normalized
        # text may omit a board, grade, location, or pending detail from history.
        answer_question = retrieval_context if followup else contextual_question

        if grade_specific_age_is_missing(retrieval_context, approved_hits):
            unanswered = self.unanswered.create(request.school_id, ticket_question)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=(
                    "I don’t have an approved grade-specific age cutoff for this "
                    "enquiry. I’ve sent the grade and your question to the school "
                    "team for confirmation. You may add your email below if you "
                    "would like to receive their answer directly."
                ),
                outcome="escalated",
                confidence=approved_hits[0].score if approved_hits else 0.0,
                unanswered_id=unanswered.id,
            )

        # Strong single-source matches are safest and fastest when returned
        # extractively.  The text is already published school knowledge, so this
        # path avoids both hallucination and two expensive local-model round trips.
        extractive_answer = extractive_fast_answer(
            answer_question,
            approved_hits,
            self.min_score,
            response_question=request.message,
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
        # Admin suggestions are always reviewed by authorized staff. Search the
        # question directly instead of spending a model round trip normalizing it.
        # Retrieval already handles domain synonyms and common spelling errors.
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
            evidence = approved_hits[:3]
            try:
                candidates = await self.model.suggest(question, evidence)
            except (KeyError, RuntimeError, TypeError, ValueError):
                candidates = []

            # One draft and one verification call keep CPU deployments responsive.
            # Additional safe options are supplied by deterministic templates.
            candidate = candidates[0] if candidates else None
            verified = False
            if candidate:
                try:
                    verified = await self.model.verify(question, candidate, evidence)
                except (KeyError, RuntimeError, TypeError, ValueError):
                    verified = False
            if candidate and verified:
                top_score = evidence[0].score
                confidence_label = "strong" if top_score >= self.min_score else "possible"
                warning = (
                    None
                    if confidence_label == "strong"
                    else (
                        "Possible knowledge match only. Confirm every detail against the "
                        "school’s current information before approval."
                    )
                )
                citations = [
                    Citation(
                        knowledge_id=hit.entry.id,
                        title=hit.entry.title,
                        source_label=hit.entry.source_label,
                        excerpt=hit.entry.content[:240],
                        score=hit.score,
                    )
                    for hit in evidence
                ]
                suggestions.append(
                    AdminAnswerSuggestion(
                        answer=candidate,
                        kind="grounded",
                        citations=citations,
                        confidence=top_score,
                        confidence_label=confidence_label,
                        warning=warning,
                    )
                )

        for template in answer_templates(question):
            if len(suggestions) >= 3:
                break
            suggestions.append(
                AdminAnswerSuggestion(
                    answer=fill_template_context(template, question),
                    kind="template",
                    requires_staff_verification=True,
                    confidence_label="none",
                    warning="Complete and verify all placeholders before approval.",
                )
            )

        if len(suggestions) < 3:
            suggestions.append(
                AdminAnswerSuggestion(
                    answer=CALLBACK_SUGGESTION,
                    kind="callback",
                    requires_staff_verification=True,
                    confidence_label="none",
                )
            )

        return AdminSuggestionResponse(
            unanswered_id=unanswered_id,
            question=question,
            suggestions=suggestions,
        )
