from app.embeddings import SentenceTransformerEmbeddingModel
from app.postgres_knowledge import ingest_entries
from app.seed import load_published_seed_data


def main() -> None:
    entries = load_published_seed_data()
    model = SentenceTransformerEmbeddingModel()
    result = ingest_entries(
        school_slug="jhs-electronic-city",
        school_name="Jain Heritage School Electronic City",
        entries=entries,
        embedding_model=model,
    )
    print(f"School ID: {result.school_id}")
    print(f"Entries written: {result.entries_written}")
    print(f"Chunks written: {result.chunks_written}")


if __name__ == "__main__":
    main()

