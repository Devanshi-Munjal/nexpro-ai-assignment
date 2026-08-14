from pathlib import Path
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from ingestion import create_chunks, load_corpus


QDRANT_PATH = Path("data/qdrant")
COLLECTION_NAME = "dunder_mifflin"

VECTOR_SIZE = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_client() -> QdrantClient:
    """Create a persistent local Qdrant client."""

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    return QdrantClient(
        path=str(QDRANT_PATH)
    )


def create_collection(client: QdrantClient) -> None:
    """Create the collection if it does not already exist."""

    existing_collections = client.get_collections()

    collection_names = {
        collection.name
        for collection in existing_collections.collections
    }

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(f"Created collection: {COLLECTION_NAME}")

    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def build_chunk_records():
    """Load the corpus and create chunk records."""

    documents = load_corpus(Path("data/corpus"))

    all_chunks = []

    for document in documents:
        chunks = create_chunks(document)
        all_chunks.extend(chunks)

    return all_chunks


def embed_chunks(model: SentenceTransformer, chunks):
    """Generate embeddings for all chunk texts."""

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


def insert_chunks(client: QdrantClient, chunks, embeddings) -> None:
    """Insert chunk vectors and metadata into Qdrant."""

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
            vector=embedding.tolist(),
            payload={
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


if __name__ == "__main__":

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Creating Qdrant client...")
    client = create_client()

    try:
        create_collection(client)

        print("Creating chunk records...")
        chunks = build_chunk_records()

        print(f"Total chunks: {len(chunks)}")

        print("Generating embeddings...")
        embeddings = embed_chunks(model, chunks)

        print("Inserting vectors into Qdrant...")
        insert_chunks(client, chunks, embeddings)

        collection_info = client.get_collection(
            collection_name=COLLECTION_NAME
        )

        print("\nIngestion complete.")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Vector size: {VECTOR_SIZE}")
        print(f"Points: {collection_info.points_count}")

    finally:
        client.close()