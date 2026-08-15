from pathlib import Path
import argparse
import json
import re

from transformers import pipeline


DEFAULT_RESULTS_PATH = Path(
    "evaluate/rag_results.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "evaluate/groundedness_metrics.txt"
)

NLI_MODEL = (
    "cross-encoder/nli-deberta-v3-small"
)

ENTAILMENT_THRESHOLD = 0.80

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


def split_into_sentences(text):
    """Split text into simple sentence-level claims."""

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def get_nli_scores(
    nli,
    premise,
    hypothesis,
):
    """Run NLI for one evidence/claim pair."""

    output = nli(
        {
            "text": premise,
            "text_pair": hypothesis,
        }
    )

    return {
        item["label"].lower(): item["score"]
        for item in output
    }


def evaluate_answer(
    nli,
    result,
):
    """Evaluate groundedness of one generated answer."""

    answer = result[
        "generated_answer"
    ]

    if (
        answer.strip()
        == REFUSAL_PHRASE
    ):

        return {
            "classification": "refusal",
            "claim_results": [],
            "grounded_claims": 0,
            "total_claims": 0,
            "unsupported_claims": 0,
        }

    # ---------------------------------------------------------
    # Evidence sentences
    # ---------------------------------------------------------

    evidence_sentences = []

    for chunk in result[
        "retrieved_chunks"
    ]:

        for sentence in split_into_sentences(
            chunk["text"]
        ):

            evidence_sentences.append(
                {
                    "chunk_id": chunk[
                        "chunk_id"
                    ],
                    "text": sentence,
                }
            )

    # ---------------------------------------------------------
    # Generated claims
    # ---------------------------------------------------------

    claims = split_into_sentences(
        answer
    )

    claim_results = []

    for claim in claims:

        best_entailment = {
            "score": 0.0,
            "chunk_id": None,
            "text": None,
        }

        for evidence in evidence_sentences:

            scores = get_nli_scores(
                nli=nli,
                premise=evidence["text"],
                hypothesis=claim,
            )

            entailment = scores.get(
                "entailment",
                0.0,
            )

            if (
                entailment
                > best_entailment["score"]
            ):

                best_entailment = {
                    "score": entailment,
                    "chunk_id": evidence[
                        "chunk_id"
                    ],
                    "text": evidence[
                        "text"
                    ],
                }

        if (
            best_entailment["score"]
            >= ENTAILMENT_THRESHOLD
        ):

            classification = "entailed"

        else:

            classification = "unsupported"

        claim_results.append(
            {
                "claim": claim,
                "classification": classification,
                "best_entailment_score": (
                    best_entailment["score"]
                ),
                "best_entailment_chunk": (
                    best_entailment["chunk_id"]
                ),
                "best_entailment_evidence": (
                    best_entailment["text"]
                ),
            }
        )

    grounded_claims = sum(
        claim["classification"]
        == "entailed"
        for claim in claim_results
    )

    unsupported_claims = sum(
        claim["classification"]
        == "unsupported"
        for claim in claim_results
    )

    total_claims = len(
        claim_results
    )

    if unsupported_claims > 0:

        answer_classification = (
            "partially_supported"
        )

    elif total_claims > 0:

        answer_classification = (
            "fully_supported"
        )

    else:

        answer_classification = (
            "unsupported"
        )

    return {
        "classification": answer_classification,
        "claim_results": claim_results,
        "grounded_claims": grounded_claims,
        "total_claims": total_claims,
        "unsupported_claims": unsupported_claims,
    }


def calculate_metrics(evaluations):

    answerable = [
        item
        for item in evaluations
        if item["answerable"]
    ]

    total_claims = sum(
        item["total_claims"]
        for item in answerable
    )

    grounded_claims = sum(
        item["grounded_claims"]
        for item in answerable
    )

    unsupported_claims = sum(
        item["unsupported_claims"]
        for item in answerable
    )

    grounded_claim_rate = (
        grounded_claims / total_claims
        if total_claims
        else 0.0
    )

    unsupported_rate = (
        unsupported_claims / total_claims
        if total_claims
        else 0.0
    )

    fully_supported = sum(
        item["classification"]
        == "fully_supported"
        for item in answerable
    )

    partially_supported = sum(
        item["classification"]
        == "partially_supported"
        for item in answerable
    )

    unanswerable = [
        item
        for item in evaluations
        if not item["answerable"]
    ]

    refused = sum(
        item["classification"]
        == "refusal"
        for item in unanswerable
    )

    return {
        "answerable_count": len(answerable),
        "total_claims": total_claims,
        "grounded_claims": grounded_claims,
        "unsupported_claims": unsupported_claims,
        "grounded_claim_rate": grounded_claim_rate,
        "unsupported_rate": unsupported_rate,
        "fully_supported": fully_supported,
        "partially_supported": partially_supported,
        "unanswerable_count": len(
            unanswerable
        ),
        "correct_refusals": refused,
    }


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RAG answer groundedness "
            "using an NLI model."
        )
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
        help="Path for the groundedness report.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("Loading evaluation results...")

    results = load_results(
        args.results
    )

    print(
        f"Total questions: {len(results)}"
    )

    print("\nLoading NLI model...")

    nli = pipeline(
        "text-classification",
        model=NLI_MODEL,
        top_k=None,
    )

    print("NLI model loaded.")

    evaluations = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        evaluation = evaluate_answer(
            nli=nli,
            result=result,
        )

        evaluations.append(
            {
                "id": result["id"],
                "question": result[
                    "question"
                ],
                "answerable": result[
                    "answerable"
                ],
                "generated_answer": result[
                    "generated_answer"
                ],
                **evaluation,
            }
        )

        print(
            f"[{index}/{len(results)}] "
            f"{result['id']}: "
            f"{evaluation['classification']}"
        )

    metrics = calculate_metrics(
        evaluations
    )

    output = f"""
DUNDER MIFFLIN RAG GROUNDEDNESS EVALUATION
============================================

NLI MODEL
--------------------------------------------
{NLI_MODEL}

ENTAILMENT THRESHOLD
--------------------------------------------
{ENTAILMENT_THRESHOLD:.2f}

ANSWERABLE QUESTIONS
--------------------------------------------
Questions: {metrics["answerable_count"]}
Claims evaluated: {metrics["total_claims"]}

Grounded claims:       {metrics["grounded_claims"]}
Unsupported claims:    {metrics["unsupported_claims"]}

Grounded Claim Rate:
{metrics["grounded_claim_rate"]:.4f} ({metrics["grounded_claim_rate"] * 100:.2f}%)

Unsupported Claim Rate:
{metrics["unsupported_rate"]:.4f} ({metrics["unsupported_rate"] * 100:.2f}%)

Fully supported answers:
{metrics["fully_supported"]}

Partially supported answers:
{metrics["partially_supported"]}

UNANSWERABLE QUESTIONS
--------------------------------------------
Total: {metrics["unanswerable_count"]}
Correct refusals: {metrics["correct_refusals"]}

METHODOLOGY
--------------------------------------------
Each generated answer is split into sentence-level claims.

Each claim is compared independently against individual
sentences extracted from the chunks actually retrieved
by the RAG pipeline.

A claim is classified as entailed when the maximum NLI
entailment score across retrieved evidence sentences is
at least {ENTAILMENT_THRESHOLD:.2f}.

Claims that do not reach the entailment threshold are
classified as unsupported.

This is an automated NLI-based groundedness proxy and
should not be interpreted as human-verified ground truth.
"""

    print("\n" + "=" * 70)
    print(output)
    print("=" * 70)

    print("\nPER-QUESTION RESULTS")
    print("=" * 70)

    for item in evaluations:

        print(
            f"\n{item['id']}: "
            f"{item['classification']}"
        )

        for claim in item[
            "claim_results"
        ]:

            print(
                f"  [{claim['classification']}] "
                f"{claim['claim']}"
            )

            print(
                f"    Entailment: "
                f"{claim['best_entailment_score']:.3f} "
                f"({claim['best_entailment_chunk']})"
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
            "\n\nPER-QUESTION RESULTS\n"
            "====================\n"
        )

        for item in evaluations:

            f.write(
                f"\n{item['id']}: "
                f"{item['classification']}\n"
            )

            for claim in item[
                "claim_results"
            ]:

                f.write(
                    f"  [{claim['classification']}] "
                    f"{claim['claim']}\n"
                )

                f.write(
                    f"    Entailment: "
                    f"{claim['best_entailment_score']:.4f} "
                    f"({claim['best_entailment_chunk']})\n"
                )

    print(
        f"\nSaved metrics to: {args.output}"
    )


if __name__ == "__main__":
    main()