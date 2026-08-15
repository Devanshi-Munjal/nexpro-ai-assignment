from pathlib import Path

from sentence_transformers import SentenceTransformer

from qdrant_client.models import Filter, FieldCondition, MatchValue

from vector_store import (
    COLLECTION_NAME,
    create_client,
)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def create_embedder():
    """Load the embedding model used during ingestion."""

    return SentenceTransformer(EMBEDDING_MODEL)


def search(
    client,
    model,
    query: str,
    k: int = TOP_K,
    document_id: str | None = None,
):
    """Search Qdrant for the most relevant chunks."""

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    query_filter = None

    if document_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
        with_vectors=False,
    ).points

    return results


if __name__ == "__main__":

    model = create_embedder()
    client = create_client()

    try:
        query = input("\nAsk Dunder Mifflin a question: ")

        document_id = input(
            "Optional document filter (press Enter for none): "
        ).strip()

        if not document_id:
            document_id = None

        results = search(
            client=client,
            model=model,
            query=query,
            k=TOP_K,
            document_id=document_id,
        )

        print(f"\nTop {len(results)} results:\n")

        for rank, result in enumerate(results, start=1):

            payload = result.payload

            print("=" * 70)
            print(f"Rank: {rank}")
            print(f"Score: {result.score:.4f}")
            print(f"Source: {payload['source']}")
            print(f"Chunk: {payload['chunk_id']}")
            print()
            print(payload["text"][:700])

    finally:
        client.close()