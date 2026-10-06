import json
from pathlib import Path

from .models import KnowledgeEntry


DATA_DIRECTORY = Path(__file__).parent / "data"
KNOWLEDGE_SEED_FILES = ("jhs_electronic_city.json",)


def load_published_seed_data() -> list[KnowledgeEntry]:
    """Load reviewed content until PostgreSQL ingestion replaces this seed."""

    entries: list[KnowledgeEntry] = []
    # Evaluation fixtures live in the same package but are not knowledge records.
    # Load only explicitly approved seed files instead of every JSON file.
    for filename in KNOWLEDGE_SEED_FILES:
        path = DATA_DIRECTORY / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries.extend(KnowledgeEntry.model_validate(item) for item in payload)
    return entries
