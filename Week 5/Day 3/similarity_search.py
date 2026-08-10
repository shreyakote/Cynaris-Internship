import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chromadb_storage")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_collection(
    "student_notes",
    embedding_function=embedding_function
)

results = collection.query(
    query_texts=["What is Artificial Intelligence?"],
    n_results=3
)

print("\nTop 3 Similar Documents:\n")

for doc in results["documents"][0]:
    print("-", doc)

print("\nMetadata Filter Example:\n")

filtered = collection.get(
    where={"topic": "AI"}
)

print("Documents with topic=AI:", len(filtered["documents"]))