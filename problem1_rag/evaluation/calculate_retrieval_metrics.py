from pathlib import Path
import argparse
import json
import math


DEFAULT_RESULTS_PATH = Path(
    "evaluation/rag_results.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "evaluation/rag_retrieval_metrics.txt"
)

DEFAULT_K = 5


def load_results(path: Path):
    """Load raw RAG evaluation results."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def reciprocal_rank(
    retrieved_ids,
    relevant_ids,
):
    """Calculate reciprocal rank of the first relevant chunk."""

    relevant_ids = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids,
    relevant_ids,
    k,
):
    """Calculate binary-relevance nDCG@k."""

    relevant_ids = set(relevant_ids)

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):

        if chunk_id in relevant_ids:

            dcg += (
                1.0
                / math.log2(rank + 1)
            )

    num_relevant = min(
        len(relevant_ids),
        k,
    )

    if num_relevant == 0:
        return 0.0

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            num_relevant + 1,
        )
    )

    return dcg / ideal_dcg


def calculate_metrics(
    results,
    k,
):
    """Calculate retrieval metrics on answerable questions."""

    answerable = [
        result
        for result in results
        if result["answerable"]
    ]

    recall_values = []
    hit_values = []
    reciprocal_rank_values = []
    ndcg_values = []
    context_precision_values = []

    for result in answerable:

        relevant_ids = result[
            "relevant_chunks"
        ]

        retrieved_ids = [
            chunk["chunk_id"]
            for chunk in result[
                "retrieved_chunks"
            ][:k]
        ]

        relevant_set = set(
            relevant_ids
        )

        retrieved_relevant = sum(
            1
            for chunk_id in retrieved_ids
            if chunk_id in relevant_set
        )

        recall = (
            retrieved_relevant
            / len(relevant_set)
            if relevant_set
            else 0.0
        )

        hit = (
            1.0
            if retrieved_relevant > 0
            else 0.0
        )

        mrr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        ndcg = calculate_ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        )

        context_precision = (
            retrieved_relevant
            / len(retrieved_ids)
            if retrieved_ids
            else 0.0
        )

        recall_values.append(recall)
        hit_values.append(hit)
        reciprocal_rank_values.append(mrr)
        ndcg_values.append(ndcg)
        context_precision_values.append(
            context_precision
        )

    question_count = len(answerable)

    if question_count == 0:
        raise ValueError(
            "No answerable questions found."
        )

    return {
        "recall_at_k": (
            sum(recall_values)
            / question_count
        ),
        "hit_rate_at_k": (
            sum(hit_values)
            / question_count
        ),
        "mrr_at_k": (
            sum(reciprocal_rank_values)
            / question_count
        ),
        "ndcg_at_k": (
            sum(ndcg_values)
            / question_count
        ),
        "context_precision_at_k": (
            sum(context_precision_values)
            / question_count
        ),
        "answerable_questions": question_count,
        "unanswerable_questions": (
            len(results)
            - question_count
        ),
    }


def parse_args():

    parser = argparse.ArgumentParser(
        description="Calculate RAG retrieval metrics."
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to rag_results.json.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the metrics report.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Retrieval cutoff.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    results = load_results(
        args.results
    )

    print(
        f"Total questions: {len(results)}"
    )

    answerable_count = sum(
        result["answerable"]
        for result in results
    )

    print(
        f"Answerable questions: "
        f"{answerable_count}"
    )

    print(
        f"Unanswerable questions: "
        f"{len(results) - answerable_count}"
    )

    print(
        f"Evaluating retrieval at k={args.k}"
    )

    metrics = calculate_metrics(
        results,
        args.k,
    )

    recall = metrics["recall_at_k"]
    hit_rate = metrics["hit_rate_at_k"]
    mrr = metrics["mrr_at_k"]
    ndcg = metrics["ndcg_at_k"]
    context_precision = (
        metrics["context_precision_at_k"]
    )

    output = f"""DUNDER MIFFLIN RAG RETRIEVAL EVALUATION
=============================================

Top-K: {args.k}

ANSWERABLE QUESTIONS
-------------------------
Total: {metrics["answerable_questions"]}

Recall@{args.k}:             {recall:.4f} ({recall * 100:.2f}%)
Hit Rate@{args.k}:            {hit_rate:.4f} ({hit_rate * 100:.2f}%)
MRR@{args.k}:                 {mrr:.4f}
nDCG@{args.k}:                {ndcg:.4f}
Context Precision@{args.k}:   {context_precision:.4f} ({context_precision * 100:.2f}%)

UNANSWERABLE QUESTIONS
-------------------------
Total: {metrics["unanswerable_questions"]}

NOTE
-------------------------
Retrieval metrics are calculated only on answerable questions
because unanswerable questions have no relevant chunk labels.

Recall@K measures the proportion of relevant chunks retrieved
within the top K.

Hit Rate@K measures whether at least one relevant chunk was
retrieved within the top K.

MRR@K measures the reciprocal rank of the first relevant chunk.

nDCG@K measures ranking quality using binary chunk relevance.

Context Precision@K measures the proportion of retrieved
top-K chunks that are relevant.
"""

    print("\n" + output)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(output)

    print(
        f"Saved metrics to: {args.output}"
    )


if __name__ == "__main__":
    main()