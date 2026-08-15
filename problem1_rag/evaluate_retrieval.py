import json
from pathlib import Path

from retrieval import create_embedder, create_client, search


EVALUATION_FILE = Path("evaluation/questions.json")
RESULTS_FILE = Path("evaluation/retrieval_results.json")
METRICS_FILE = Path("evaluation/retrieval_metrics.txt")

TOP_K = 5


def load_questions():
    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_question(client, model, question):
    results = search(
        client=client,
        model=model,
        query=question["question"],
        k=TOP_K,
    )

    retrieved_chunks = [
        result.payload["chunk_id"]
        for result in results
    ]

    relevant_chunks = set(question["relevant_chunks"])

    # Only calculate Hit@K for answerable questions.
    hit = None

    if question["answerable"]:
        hit = bool(
            relevant_chunks.intersection(retrieved_chunks)
        )

    return {
        "question_id": question["id"],
        "question": question["question"],
        "answerable": question["answerable"],
        "relevant_chunks": question["relevant_chunks"],
        "retrieved_chunks": retrieved_chunks,
        "hit_at_k": hit,
    }


def calculate_metrics(results):
    answerable_results = [
        result
        for result in results
        if result["answerable"]
    ]

    unanswerable_results = [
        result
        for result in results
        if not result["answerable"]
    ]

    answerable_hits = sum(
        result["hit_at_k"]
        for result in answerable_results
    )

    answerable_total = len(answerable_results)

    answerable_misses = (
        answerable_total - answerable_hits
    )

    hit_rate = (
        answerable_hits / answerable_total
        if answerable_total
        else 0
    )

    return {
        "total_questions": len(results),
        "answerable_questions": answerable_total,
        "answerable_hits": answerable_hits,
        "answerable_misses": answerable_misses,
        "answerable_hit_at_k": hit_rate,
        "unanswerable_questions": len(unanswerable_results),
    }


def save_results(results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_metrics(metrics):
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write("DUNDER MIFFLIN RAG RETRIEVAL EVALUATION\n")
        f.write("=" * 45 + "\n\n")

        f.write(f"Top-K: {TOP_K}\n\n")

        f.write("ANSWERABLE QUESTIONS\n")
        f.write("-" * 25 + "\n")
        f.write(
            f"Total:  {metrics['answerable_questions']}\n"
        )
        f.write(
            f"Hits:   {metrics['answerable_hits']}\n"
        )
        f.write(
            f"Misses: {metrics['answerable_misses']}\n"
        )
        f.write(
            f"Hit@{TOP_K}: "
            f"{metrics['answerable_hit_at_k']:.4f} "
            f"({metrics['answerable_hit_at_k'] * 100:.2f}%)\n\n"
        )

        f.write("UNANSWERABLE QUESTIONS\n")
        f.write("-" * 25 + "\n")
        f.write(
            f"Total: {metrics['unanswerable_questions']}\n"
        )

        f.write("\n")
        f.write("NOTE\n")
        f.write("-" * 25 + "\n")
        f.write(
            "Hit@K is calculated only on answerable questions. "
            "Unanswerable questions are evaluated separately "
            "during end-to-end RAG evaluation.\n"
        )


if __name__ == "__main__":

    questions = load_questions()

    print(
        f"Loaded {len(questions)} evaluation questions."
    )
    print(
        f"Evaluating retrieval with k={TOP_K}\n"
    )

    model = create_embedder()
    client = create_client()

    try:
        results = []

        for i, question in enumerate(
            questions,
            start=1,
        ):
            print(
                f"[{i}/{len(questions)}] "
                f"{question['id']}: "
                f"{question['question']}"
            )

            result = evaluate_question(
                client,
                model,
                question,
            )

            results.append(result)

            if result["answerable"]:
                print(
                    f"  Hit@{TOP_K}: "
                    f"{result['hit_at_k']}"
                )
            else:
                print(
                    "  Unanswerable: "
                    "not included in Hit@K"
                )

            print(
                f"  Retrieved: "
                f"{result['retrieved_chunks']}"
            )
            print()

        metrics = calculate_metrics(results)

        save_results(results)
        save_metrics(metrics)

        print("=" * 70)
        print("RETRIEVAL EVALUATION")
        print("=" * 70)

        print(
            f"Answerable questions: "
            f"{metrics['answerable_questions']}"
        )
        print(
            f"Hits:                 "
            f"{metrics['answerable_hits']}"
        )
        print(
            f"Misses:               "
            f"{metrics['answerable_misses']}"
        )
        print(
            f"Hit@{TOP_K}:              "
            f"{metrics['answerable_hit_at_k']:.4f} "
            f"({metrics['answerable_hit_at_k'] * 100:.2f}%)"
        )
        print(
            f"Unanswerable questions: "
            f"{metrics['unanswerable_questions']}"
        )

        print("\nSaved:")
        print(f"  {RESULTS_FILE}")
        print(f"  {METRICS_FILE}")

    finally:
        client.close()