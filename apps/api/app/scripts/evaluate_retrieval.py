import json
from pathlib import Path

from app.embeddings import SentenceTransformerEmbeddingModel
from app.postgres_knowledge import search_entries
from app.retrieval_policy import evaluate_evidence, expand_query


EVALUATION_FILE = Path(__file__).parents[1] / "data" / "retrieval_evaluation.json"


def main() -> None:
    cases = json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    model = SentenceTransformerEmbeddingModel()
    top1_correct = 0
    top3_correct = 0
    unsupported_correct = 0
    unsupported_total = 0

    for case in cases:
        question = case["question"]
        expected = case.get("expected_title")
        expected_titles = case.get("expected_titles")
        supported_titles = expected_titles or ([expected] if expected else [])
        expanded = expand_query(question)
        vector = model.embed([expanded])[0]
        results = search_entries("jhs-electronic-city", question, vector, limit=5)
        supported_results = []
        for result in results:
            result_decision = evaluate_evidence(
                question,
                f"{result['title']} {result['content']} {' '.join(result['tags'])}",
                float(result["combined_score"]),
            )
            if result_decision.supported:
                supported_results.append(result)

        top = supported_results[0] if supported_results else None
        raw_top = results[0] if results else None

        if not supported_titles:
            unsupported_total += 1
            passed = not supported_results
            unsupported_correct += int(passed)
        else:
            titles = [result["title"] for result in supported_results]
            top1_correct += int(bool(top) and top["title"] in supported_titles)
            top3_correct += int(any(title in titles[:3] for title in supported_titles))
            passed = bool(top) and top["title"] in supported_titles

        actual = top["title"] if top else (raw_top["title"] if raw_top else "NO RESULT")
        action = "ANSWER" if top else "ESCALATE"
        marker = "PASS" if passed else "FAIL"
        expected_label = " / ".join(supported_titles) or "UNSUPPORTED"
        print(f"{marker:4} | {action:8} | expected={expected_label} | actual={actual} | {question}")

    supported_total = len(cases) - unsupported_total
    print("\nSUMMARY")
    print(f"Supported Top-1: {top1_correct}/{supported_total}")
    print(f"Supported Top-3: {top3_correct}/{supported_total}")
    print(f"Unsupported rejected: {unsupported_correct}/{unsupported_total}")


if __name__ == "__main__":
    main()
