from pathlib import Path
import argparse
import json
import time

from sentence_transformers import SentenceTransformer

from retrieval import search
from vector_store import create_client, COLLECTION_NAME
from generation import load_model, generate_answer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

DEFAULT_QUESTIONS_PATH = Path(
    "evaluation/questions.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "evaluation/rag_results.json"
)


def load_questions(path: Path):
    """Load the fixed evaluation question set."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def evaluate(
    questions,
    client,
    embedding_model,
    tokenizer,
    llm_model,
    top_k,
):
    """Run the RAG pipeline over the evaluation questions."""

    results = []

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item["id"]
        question = item["question"]

        print("\n" + "=" * 70)
        print(
            f"[{index}/{len(questions)}] "
            f"{question_id}"
        )
        print(question)

        total_start = time.perf_counter()

        # ---------------------------------------------------------
        # Retrieval
        # ---------------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieved = search(
            client=client,
            model=embedding_model,
            query=question,
            k=top_k,
        )

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        retrieved_chunks = [
            {
                "chunk_id": result.payload["chunk_id"],
                "document_id": result.payload["document_id"],
                "source": result.payload["source"],
                "score": result.score,
                "text": result.payload["text"],
            }
            for result in retrieved
        ]

        # ---------------------------------------------------------
        # Context construction
        # ---------------------------------------------------------

        context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        # ---------------------------------------------------------
        # Generation
        # ---------------------------------------------------------

        generation_start = time.perf_counter()

        answer = generate_answer(
            tokenizer=tokenizer,
            model=llm_model,
            question=question,
            context=context,
        )

        generation_latency_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        # ---------------------------------------------------------
        # Token usage
        # ---------------------------------------------------------

        input_tokens = len(
            tokenizer.encode(
                context,
                add_special_tokens=False,
            )
        )

        output_tokens = len(
            tokenizer.encode(
                answer,
                add_special_tokens=False,
            )
        )

        total_tokens = (
            input_tokens
            + output_tokens
        )

        # ---------------------------------------------------------
        # Save result
        # ---------------------------------------------------------

        result = {
            "id": question_id,
            "question": question,
            "answerable": item["answerable"],
            "relevant_chunks": item["relevant_chunks"],
            "gold_answer": item["gold_answer"],
            "retrieved_chunks": retrieved_chunks,
            "generated_answer": answer,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        results.append(result)

        print(
            "Retrieved:",
            [
                chunk["chunk_id"]
                for chunk in retrieved_chunks
            ],
        )

        print(f"Answer: {answer}")

        print(
            f"Latency: "
            f"{total_latency_ms:.2f} ms "
            f"(retrieval: "
            f"{retrieval_latency_ms:.2f} ms, "
            f"generation: "
            f"{generation_latency_ms:.2f} ms)"
        )

        print(
            f"Tokens: "
            f"input={input_tokens}, "
            f"output={output_tokens}, "
            f"total={total_tokens}"
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the Dunder Mifflin RAG pipeline."
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to evaluation questions JSON.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the raw RAG evaluation results.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading evaluation questions...")

    questions = load_questions(
        args.questions
    )

    print(
        f"Loaded {len(questions)} "
        "evaluation questions."
    )

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("\nOpening Qdrant...")

    client = create_client()

    try:

        print("\nLoading generation model...")

        tokenizer, llm_model = load_model()

        print("\nAll components loaded successfully.")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Top-K: {args.top_k}")
        print(f"Questions: {len(questions)}")

        results = evaluate(
            questions=questions,
            client=client,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            llm_model=llm_model,
            top_k=args.top_k,
        )

    finally:
        client.close()

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(
        f"Results saved to: {args.output}"
    )
    print(
        f"Questions evaluated: {len(results)}"
    )


if __name__ == "__main__":
    main()