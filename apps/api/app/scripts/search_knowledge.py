import argparse

from app.embeddings import SentenceTransformerEmbeddingModel
from app.postgres_knowledge import search_entries
from app.retrieval_policy import evaluate_evidence, expand_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Search published JHS knowledge")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--limit", type=int, default=5)
    arguments = parser.parse_args()

    model = SentenceTransformerEmbeddingModel()
    expanded_query = expand_query(arguments.query)
    query_embedding = model.embed([expanded_query])[0]
    results = search_entries(
        school_slug="jhs-electronic-city",
        query=arguments.query,
        query_embedding=query_embedding,
        limit=arguments.limit,
    )

    if not results:
        print("No published knowledge found.")
        return

    accepted = []
    for result in results:
        result_decision = evaluate_evidence(
            arguments.query,
            f"{result['title']} {result['content']} {' '.join(result['tags'])}",
            float(result["combined_score"]),
        )
        if result_decision.supported:
            accepted.append((result, result_decision))

    top = accepted[0][0] if accepted else results[0]
    decision = accepted[0][1] if accepted else evaluate_evidence(
        arguments.query,
        f"{top['title']} {top['content']} {' '.join(top['tags'])}",
        float(top["combined_score"]),
    )
    print(f"Expanded query: {expanded_query}")
    print(f"Evidence decision: {'ANSWER' if decision.supported else 'ESCALATE'}")
    print(f"Reason: {decision.reason}")

    for index, result in enumerate(results, start=1):
        print(f"\n{index}. {result['title']}")
        print(f"   Combined: {result['combined_score']:.4f}")
        print(f"   Semantic: {result['semantic_score']:.4f}")
        print(f"   Keyword: {result['keyword_score']:.4f}")
        print(f"   Synonym: {result['synonym_score']:.4f}")
        print(f"   Source: {result['source_label']}")
        print(f"   Text: {result['content']}")


if __name__ == "__main__":
    main()
