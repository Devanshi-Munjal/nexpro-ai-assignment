from pathlib import Path
from collections import Counter
import argparse
import json
import re
import string


DEFAULT_RESULTS_PATH = Path(
    "evaluation/rag_results.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "evaluation/answer_metrics.txt"
)

REFUSAL_PHRASE = (
    "I don't have enough information in the provided "
    "company policies to answer that."
)


def load_results(path: Path):
    """Load raw RAG evaluation results."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def normalize_text(text):
    """Normalize text for EM and token-level F1."""

    text = text.lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def exact_match(
    prediction,
    gold,
):
    """Calculate normalized exact match."""

    return int(
        normalize_text(prediction)
        == normalize_text(gold)
    )


def token_f1(
    prediction,
    gold,
):
    """Calculate token-level F1."""

    prediction_tokens = normalize_text(
        prediction
    ).split()

    gold_tokens = normalize_text(
        gold
    ).split()

    if not prediction_tokens and not gold_tokens:
        return 1.0

    if not prediction_tokens or not gold_tokens:
        return 0.0

    prediction_counts = Counter(
        prediction_tokens
    )

    gold_counts = Counter(
        gold_tokens
    )

    common = (
        prediction_counts
        & gold_counts
    )

    num_common = sum(
        common.values()
    )

    if num_common == 0:
        return 0.0

    precision = (
        num_common
        / len(prediction_tokens)
    )

    recall = (
        num_common
        / len(gold_tokens)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def is_refusal(answer):
    """Check whether the answer contains the expected refusal."""

    normalized_answer = normalize_text(
        answer
    )

    normalized_refusal = normalize_text(
        REFUSAL_PHRASE
    )

    return (
        normalized_refusal
        in normalized_answer
    )


def calculate_metrics(results):

    answerable = [
        result
        for result in results
        if result["answerable"]
    ]

    unanswerable = [
        result
        for result in results
        if not result["answerable"]
    ]

    em_scores = []
    f1_scores = []
    per_question = []

    for result in answerable:

        prediction = result[
            "generated_answer"
        ]

        gold = result[
            "gold_answer"
        ]

        em = exact_match(
            prediction,
            gold,
        )

        f1 = token_f1(
            prediction,
            gold,
        )

        em_scores.append(em)
        f1_scores.append(f1)

        per_question.append(
            {
                "id": result["id"],
                "exact_match": em,
                "token_f1": f1,
            }
        )

    if not answerable:
        raise ValueError(
            "No answerable questions found."
        )

    mean_em = (
        sum(em_scores)
        / len(answerable)
    )

    mean_f1 = (
        sum(f1_scores)
        / len(answerable)
    )

    refusal_results = []

    for result in unanswerable:

        refused = is_refusal(
            result["generated_answer"]
        )

        refusal_results.append(
            {
                "id": result["id"],
                "refused": refused,
            }
        )

    refusal_accuracy = (
        sum(
            item["refused"]
            for item in refusal_results
        )
        / len(unanswerable)
        if unanswerable
        else 0.0
    )

    return {
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "exact_match": mean_em,
        "token_f1": mean_f1,
        "refusal_accuracy": refusal_accuracy,
        "per_question": per_question,
        "refusal_results": refusal_results,
    }


def parse_args():

    parser = argparse.ArgumentParser(
        description="Calculate RAG answer metrics."
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

    return parser.parse_args()


def main():

    args = parse_args()

    results = load_results(
        args.results
    )

    metrics = calculate_metrics(
        results
    )

    output = f"""DUNDER MIFFLIN RAG ANSWER EVALUATION
=========================================

ANSWERABLE QUESTIONS
-------------------------
Total: {metrics["answerable_count"]}

Exact Match (EM):       {metrics["exact_match"]:.4f} ({metrics["exact_match"] * 100:.2f}%)
Token F1:               {metrics["token_f1"]:.4f} ({metrics["token_f1"] * 100:.2f}%)

UNANSWERABLE QUESTIONS
-------------------------
Total: {metrics["unanswerable_count"]}

Refusal Accuracy:       {metrics["refusal_accuracy"]:.4f} ({metrics["refusal_accuracy"] * 100:.2f}%)

NOTE
-------------------------
EM requires the normalized generated answer to exactly
match the normalized gold answer.

Token F1 measures token-level overlap between the generated
answer and gold answer.

EM/F1 do not fully capture valid paraphrases or semantic
equivalence, so they are not treated as the sole measure
of answer quality.

Refusal accuracy measures whether unanswerable questions
received the expected abstention response.
"""

    print(output)

    print("=" * 70)
    print("PER-QUESTION ANSWER SCORES")
    print("=" * 70)

    for item in metrics["per_question"]:

        print(
            f"{item['id']}: "
            f"EM={item['exact_match']} "
            f"F1={item['token_f1']:.4f}"
        )

    print("\n" + "=" * 70)
    print("UNANSWERABLE / REFUSAL SCORES")
    print("=" * 70)

    for item in metrics[
        "refusal_results"
    ]:

        print(
            f"{item['id']}: "
            f"refused={item['refused']}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(output)

        f.write(
            "\n\nPER-QUESTION SCORES\n"
            "====================\n"
        )

        for item in metrics[
            "per_question"
        ]:

            f.write(
                f"{item['id']}: "
                f"EM={item['exact_match']} "
                f"F1={item['token_f1']:.4f}\n"
            )

        f.write(
            "\nUNANSWERABLE / REFUSAL SCORES\n"
            "==============================\n"
        )

        for item in metrics[
            "refusal_results"
        ]:

            f.write(
                f"{item['id']}: "
                f"refused={item['refused']}\n"
            )

    print(
        f"\nSaved metrics to: {args.output}"
    )


if __name__ == "__main__":
    main()