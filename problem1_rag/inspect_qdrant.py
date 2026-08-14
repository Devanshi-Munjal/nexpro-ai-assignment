from qdrant_client import QdrantClient

from vector_store import (
    COLLECTION_NAME,
    QDRANT_PATH,
    create_client,
)


if __name__ == "__main__":
    client = create_client()

    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            print("No points found.")
        else:
            point = points[0]

            print("Point ID:")
            print(point.id)

            print("\nPayload:")
            print(point.payload)

    finally:
        client.close()