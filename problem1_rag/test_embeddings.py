from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


model = SentenceTransformer(MODEL_NAME)

texts = [
    "Dunder Mifflin employees receive 18 days of annual leave.",
    "Business travel requires manager approval before booking.",
]

embeddings = model.encode(texts)

print("Number of texts:", len(texts))
print("Embedding shape:", embeddings.shape)
print("Embedding dimension:", embeddings.shape[1])
print("First vector, first 10 values:")
print(embeddings[0][:10])