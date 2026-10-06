import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .retrieval import STOP_WORDS, TOKEN_RE, normalize_token, tokenize


DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "aquatic": ("swimming", "water sports", "pool"),
    "campus": ("location", "address"),
    "located": ("location", "address"),
    "food": ("meal", "lunch", "Sodexo"),
    "lunch": ("meal", "food", "Sodexo"),
    "class": ("grade",),
    "board": ("curriculum", "CBSE", "ICSE", "syllabus"),
    "cost": ("fee", "fees", "payment"),
    "price": ("fee", "fees", "payment"),
    "grow": ("add grade", "open grade", "reach grade 12", "growth roadmap"),
    "age": ("eligibility", "date of birth", "cut-off"),
    "assessment": ("tests", "projects", "learning portfolio", "evaluation"),
    "apply": ("application", "admission process", "enrolment"),
    "creative": ("art", "music", "movement"),
    "marks": ("whole child", "rank", "examinations"),
    "entrance": ("interaction", "admission assessment", "test"),
    "email": ("contact", "email address", "info@jhselectroniccity.com"),
}

DOMAIN_VOCABULARY = {
    "admission", "badminton", "basketball", "cbse", "child", "class",
    "contact", "cricket", "curriculum", "dance", "fee", "finnish",
    "food", "grade", "icse", "library", "location", "meal", "montessori",
    "music", "play", "playing", "school", "sports", "syllabus", "swimming",
    "transport", "yoga",
}


REQUIRED_EVIDENCE: dict[str, dict[str, set[str]]] = {
    "fees": {
        "triggers": {"fee", "fees", "cost", "price", "payment"},
        "evidence": {"fee", "fees", "cost", "price", "payment", "tuition"},
    },
    "transport": {
        "triggers": {"transport", "transportation", "bus", "buses"},
        "evidence": {"transport", "transportation", "bus", "buses", "route"},
    },
    "timings": {
        "triggers": {"timing", "timings", "hours", "schedule", "open", "close"},
        "evidence": {"timing", "timings", "hours", "schedule", "am", "pm"},
    },
    "deadlines": {
        "triggers": {"deadline", "deadlines", "last date"},
        "evidence": {"deadline", "deadlines", "last date", "closing date"},
    },
    "seat availability": {
        "triggers": {"seat", "seats", "vacancy", "vacancies", "availability"},
        "evidence": {"seat", "seats", "vacancy", "vacancies", "available"},
        "deferrals": {"must be confirmed"},
    },
    "aquatic facilities": {
        "triggers": {"aquatic", "swimming", "pool"},
        "evidence": {"aquatic", "swimming", "pool", "water sports"},
    },
    "curriculum": {
        "triggers": {"curriculum", "syllabus", "board", "cbse", "icse"},
        "evidence": {"curriculum", "syllabus", "board", "cbse", "icse"},
    },
    "email contact": {
        "triggers": {"email", "e-mail", "inbox"},
        "evidence": {"email", "e-mail", "info", "jhselectroniccity.com"},
    },
    "enquiry requirements": {
        "triggers": {"details", "information", "needed", "required"},
        "evidence": {"name", "birth", "grade", "number", "phone", "contact"},
    },
}


def normalized_phrases(text: str) -> set[str]:
    lowered = text.lower()
    phrases = set(tokenize(lowered))
    phrases.update(
        phrase
        for phrase in ("last date", "closing date", "water sports")
        if phrase in lowered
    )
    return phrases


def expand_query(query: str) -> str:
    terms = normalized_phrases(query)
    additions: list[str] = []
    for trigger, synonyms in DOMAIN_SYNONYMS.items():
        if trigger == "campus" and "visit" in terms:
            continue
        if trigger in terms:
            additions.extend(synonyms)
    if "enquiry" in terms and {"detail", "details", "information", "needed"} & terms:
        additions.extend(("child name", "date of birth", "grade", "contact number"))
    for raw_token in TOKEN_RE.findall(query.lower()):
        if len(raw_token) < 4:
            continue
        closest = max(
            DOMAIN_VOCABULARY,
            key=lambda candidate: SequenceMatcher(None, raw_token, candidate).ratio(),
        )
        if (
            raw_token != closest
            and SequenceMatcher(None, raw_token, closest).ratio() >= 0.84
        ):
            additions.append(closest)
    if not additions:
        return query
    return f"{query} Related search terms: {' '.join(dict.fromkeys(additions))}"


def synonym_evidence_score(query: str, passage: str) -> float:
    query_terms = normalized_phrases(query)
    passage_terms = normalized_phrases(passage)
    for trigger, synonyms in DOMAIN_SYNONYMS.items():
        if trigger in query_terms and normalized_phrases(" ".join(synonyms)) & passage_terms:
            return 1.0
    return 0.0


@dataclass(frozen=True)
class EvidenceDecision:
    supported: bool
    reason: str


def evaluate_evidence(
    question: str,
    passage: str,
    score: float,
    minimum_score: float = 0.38,
) -> EvidenceDecision:
    question_terms = normalized_phrases(question)
    passage_terms = normalized_phrases(passage)

    lowered_question = question.lower()
    lowered_passage = passage.lower()
    policy_question_terms = question_terms | typo_tolerant_query_tokens(question)

    if ({"timing", "timings", "hours"} & question_terms) and not re.search(
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", lowered_passage
    ):
        return EvidenceDecision(False, "retrieved passage has no approved clock-time evidence")

    if "who" in lowered_question and {"teacher", "faculty"} & question_terms:
        return EvidenceDecision(False, "no approved named staff evidence is available")

    if "after" in question_terms and {"visit", "consult"} & question_terms:
        if not {"apply", "application", "document", "interaction", "enrolment"} & passage_terms:
            return EvidenceDecision(False, "retrieved passage does not describe the next admission step")

    for topic, policy in REQUIRED_EVIDENCE.items():
        trigger_detected = bool(policy_question_terms & policy["triggers"])
        if trigger_detected:
            if not passage_terms & policy["evidence"]:
                return EvidenceDecision(False, f"retrieved passage has no approved {topic} evidence")
            if any(
                phrase in lowered_passage
                for phrase in policy.get("deferrals", set())
            ):
                return EvidenceDecision(
                    False,
                    f"approved source defers {topic} confirmation to the admissions team",
                )

    if score < minimum_score:
        keyword_score = meaningful_keyword_score(question, passage)
        strong_direct_match = score >= 0.25 and keyword_score >= 0.66
        balanced_match = score >= 0.32 and keyword_score >= 0.50
        if strong_direct_match or balanced_match:
            return EvidenceDecision(
                True,
                "strong direct or typo-tolerant evidence match",
            )
        return EvidenceDecision(False, f"retrieval score {score:.4f} is below {minimum_score:.2f}")

    return EvidenceDecision(True, "retrieved passage contains the required evidence")


def meaningful_keyword_score(query: str, passage: str) -> float:
    query_terms = typo_tolerant_query_tokens(query)
    if not query_terms:
        return 0.0
    passage_terms = tokenize(passage)
    matched = sum(
        1
        for query_term in query_terms
        if query_term in passage_terms
        or any(
            len(query_term) >= 4
            and len(passage_term) >= 4
            and SequenceMatcher(None, query_term, passage_term).ratio() >= 0.84
            for passage_term in passage_terms
        )
    )
    return matched / len(query_terms)


def typo_tolerant_query_tokens(query: str) -> set[str]:
    """Normalize query tokens and discard close misspellings of stop words."""

    terms: set[str] = set()
    for raw_token in TOKEN_RE.findall(query.lower()):
        if raw_token in STOP_WORDS:
            continue
        if len(raw_token) >= 4 and any(
            len(stop_word) >= 4
            and SequenceMatcher(None, raw_token, stop_word).ratio() >= 0.84
            for stop_word in STOP_WORDS
        ):
            continue
        closest = max(
            DOMAIN_VOCABULARY,
            key=lambda candidate: SequenceMatcher(None, raw_token, candidate).ratio(),
        )
        corrected = (
            closest
            if raw_token != closest
            and SequenceMatcher(None, raw_token, closest).ratio() >= 0.80
            else raw_token
        )
        terms.add(normalize_token(corrected))
    return terms
