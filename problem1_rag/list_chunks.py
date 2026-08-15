from vector_store import COLLECTION_NAME, create_client


if __name__ == "__main__":
    client = create_client()

    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        points.sort(
            key=lambda point: (
                point.payload["source"],
                point.payload["chunk_index"],
            )
        )

        print(f"Total points: {len(points)}\n")

        for point in points:
            payload = point.payload

            print("=" * 80)
            print(f"Chunk ID:     {payload['chunk_id']}")
            print(f"Source:       {payload['source']}")
            print(f"Document ID:  {payload['document_id']}")
            print(f"Chunk index:  {payload['chunk_index']}")
            print("-" * 80)
            print(payload["text"])
            print()

    finally:
        client.close()