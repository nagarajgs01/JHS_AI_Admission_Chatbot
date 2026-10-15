import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .retrieval import STOP_WORDS, TOKEN_RE, normalize_token, tokenize


DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "aquatic": ("swimming", "water sports", "pool"),
    "campus": ("location", "address"),
    "located": ("location", "address"),
    "state": ("location", "address", "Karnataka", "Bengaluru"),
    "city": ("location", "address", "Bengaluru", "Rayasandra"),
    "visit": ("campus visit", "booking", "appointment", "contact hours"),
    "tour": ("campus visit", "booking", "appointment"),
    "participate": ("sports", "activities", "play"),
    "food": ("meal", "lunch", "Sodexo"),
    "lunch": ("meal", "food", "Sodexo"),
    "class": ("grade",),
    "board": ("curriculum", "CBSE", "ICSE", "syllabus"),
    "cost": ("fee", "fees", "payment"),
    "price": ("fee", "fees", "payment"),
    "grow": ("add grade", "open grade", "reach grade 12", "growth roadmap"),
    "age": ("eligibility", "date of birth", "cut-off"),
    "assessment": ("tests", "projects", "learning portfolio", "evaluation"),
    "creative": ("art", "music", "movement"),
    "marks": ("whole child", "rank", "examinations"),
    "entrance": ("interaction", "admission assessment", "test"),
    "email": ("contact", "email address", "info@jhselectroniccity.com"),
    "study": ("academics", "grades", "curriculum", "board", "CBSE", "ICSE"),
    "transport": ("bus", "route", "pickup", "drop-off"),
    "transportation": ("transport", "bus", "route", "pickup", "drop-off"),
    "syllabus": ("curriculum", "board", "CBSE", "ICSE"),
    "curriculum": ("syllabus", "board", "CBSE", "ICSE"),
    "learn": ("learning", "teaching", "education"),
    "benefit": ("skills", "development", "learning outcomes"),
    "ability": ("skills", "development", "learning outcomes"),
}

DOMAIN_VOCABULARY = {
    "admission", "badminton", "basketball", "board", "cbse", "child", "class",
    "contact", "cricket", "curriculum", "dance", "fee", "finnish",
    "follow", "food", "grade", "icse", "library", "location", "meal", "montessori",
    "music", "option", "play", "playing", "school", "sports", "study", "syllabus", "swimming",
    "transport", "transportation", "bus", "buses", "route", "yoga",
    "apply", "chairman", "facility", "facilities", "state", "city",
    "karnataka", "bengaluru", "rayasandra", "country", "learn", "learning",
    "children", "kids", "benefit", "benefits", "ability", "abilities", "skills",
}

COMMON_CHAT_NORMALIZATIONS = {
    "u": "you",
    "ur": "your",
    "hw": "how",
    "hav": "have",
    "kidz": "children",
    "lern": "learn",
    "thru": "through",
    "du": "do",
    "wil": "will",
    "pik": "pick",
    "childrn": "children",
    "plaing": "playing",
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
        "triggers": {"timing", "timings", "hours", "schedule"},
        "evidence": {"timing", "timings", "hours", "schedule", "am", "pm"},
    },
    "deadlines": {
        "triggers": {"deadline", "deadlines", "last date"},
        "evidence": {"deadline", "deadlines", "last date", "closing date"},
        "deferrals": {"must be confirmed"},
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


def correct_query_typos(query: str) -> str:
    """Correct confident domain misspellings before semantic expansion."""

    corrected_query = query
    for shorthand, replacement in COMMON_CHAT_NORMALIZATIONS.items():
        def replace_shorthand(match, value=replacement) -> str:
            original = match.group(0)
            return value.capitalize() if original[:1].isupper() else value

        corrected_query = re.sub(
            rf"\b{re.escape(shorthand)}\b",
            replace_shorthand,
            corrected_query,
            flags=re.IGNORECASE,
        )

    def replace(match) -> str:
        original = match.group(0)
        raw_token = original.lower()

        def preserve_case(value: str) -> str:
            if original.isupper():
                return value.upper()
            if original[:1].isupper():
                return value.capitalize()
            return value

        if raw_token in COMMON_CHAT_NORMALIZATIONS:
            return preserve_case(COMMON_CHAT_NORMALIZATIONS[raw_token])
        if raw_token in STOP_WORDS or raw_token in DOMAIN_VOCABULARY:
            return original
        if len(raw_token) < 4:
            return original
        closest = max(
            DOMAIN_VOCABULARY,
            key=lambda candidate: SequenceMatcher(None, raw_token, candidate).ratio(),
        )
        if SequenceMatcher(None, raw_token, closest).ratio() >= 0.80:
            return preserve_case(closest)
        return original

    return TOKEN_RE.sub(replace, corrected_query)


def expand_query(query: str) -> str:
    corrected_query = correct_query_typos(query)
    terms = normalized_phrases(corrected_query)
    additions: list[str] = []
    for trigger, synonyms in DOMAIN_SYNONYMS.items():
        if trigger == "campus" and "visit" in terms:
            continue
        if trigger in terms:
            additions.extend(synonyms)
    # "Options" has no universal school-domain meaning. Expand it only from the
    # surrounding topic instead of forcing every options question toward curriculum.
    if "option" in terms:
        if terms & {"study", "board", "curriculum", "syllabus", "academic"}:
            additions.extend(("grades", "curriculum", "board", "CBSE", "ICSE"))
        elif terms & {"transport", "bus", "route"}:
            additions.extend(("bus", "route", "pickup", "drop-off"))
    # "Apply" may mean applying knowledge or submitting an admission form. Only
    # expand it toward admissions when the surrounding words establish that intent.
    learning_application_context = bool(
        terms & {"reflect", "learn", "knowledge", "concept", "lesson", "practice"}
    )
    if "apply" in terms and (
        terms & {"admission", "application", "enrolment", "enquiry", "form", "join"}
        or not learning_application_context
    ):
        additions.extend(("application", "admission process", "enrolment"))
    if terms & {"choose", "select", "compare"} and terms & {
        "board", "curriculum", "cbse", "icse"
    }:
        additions.extend(("board selection", "board counselling", "CBSE", "ICSE"))
    if "visit" in terms and terms & {"book", "appointment", "schedule"}:
        additions.extend(("campus visit", "booked in advance", "contact", "appointment"))
    if "enquiry" in terms and {"detail", "details", "information", "needed"} & terms:
        additions.extend(("child name", "date of birth", "grade", "contact number"))
    for raw_token in TOKEN_RE.findall(corrected_query.lower()):
        if len(raw_token) < 4:
            continue
        closest = max(
            DOMAIN_VOCABULARY,
            key=lambda candidate: SequenceMatcher(None, raw_token, candidate).ratio(),
        )
        if (
            raw_token != closest
            and SequenceMatcher(None, raw_token, closest).ratio() >= 0.80
        ):
            additions.append(closest)
    if not additions:
        return corrected_query
    return f"{corrected_query} Related search terms: {' '.join(dict.fromkeys(additions))}"


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

    # Office/contact hours do not establish the student school-day schedule.
    if (
        "school" in TOKEN_RE.findall(lowered_question)
        and {"timing", "timings", "schedule"} & question_terms
        and not any(
            phrase in lowered_passage
            for phrase in ("school-day timing", "school day timing", "student timing", "class timing")
        )
    ):
        return EvidenceDecision(False, "office hours do not establish student school timings")

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
        high_coverage_lexical_match = score >= 0.18 and keyword_score >= 0.75
        strong_direct_match = score >= 0.25 and keyword_score >= 0.66
        balanced_match = score >= 0.32 and keyword_score >= 0.50
        if high_coverage_lexical_match or strong_direct_match or balanced_match:
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
        raw_token = COMMON_CHAT_NORMALIZATIONS.get(raw_token, raw_token)
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


def has_likely_typo(query: str) -> bool:
    for raw_token in TOKEN_RE.findall(query.lower()):
        if len(raw_token) < 4 or raw_token in STOP_WORDS:
            continue
        # Ordinary plural forms are not spelling mistakes. Keep typo correction
        # available for genuine misspellings while avoiding an expensive LLM
        # understanding call for words such as boards, fees, classes and buses.
        singular_candidates = set()
        if raw_token.endswith("s") and len(raw_token) > 4:
            simple_s_base = raw_token[:-1]
            # A known word that already ends in "s" (for example syllabus)
            # does not form its plural by adding only another "s". This keeps
            # "syllabuss" classified as a typo while accepting board→boards.
            if not simple_s_base.endswith("s"):
                singular_candidates.add(simple_s_base)
        if raw_token.endswith("es") and len(raw_token) > 5:
            singular_candidates.add(raw_token[:-2])
        if raw_token.endswith("ies") and len(raw_token) > 5:
            singular_candidates.add(f"{raw_token[:-3]}y")
        if singular_candidates & DOMAIN_VOCABULARY:
            continue
        # Known grammatical forms such as options→option, fees→fee and
        # transportation→transport are normalization, not spelling mistakes.
        normalized_token = normalize_token(raw_token)
        if normalized_token != raw_token:
            continue
        closest = max(
            DOMAIN_VOCABULARY,
            key=lambda candidate: SequenceMatcher(None, raw_token, candidate).ratio(),
        )
        if (
            raw_token != closest
            and SequenceMatcher(None, raw_token, closest).ratio() >= 0.80
        ):
            return True
    return False
