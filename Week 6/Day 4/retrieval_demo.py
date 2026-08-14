import chromadb
from chromadb.utils import embedding_functions


# --------------------------------------------------
# CONNECT TO CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_collection(
    name="rag_documents",
    embedding_function=embedding_function
)


# --------------------------------------------------
# VERIFY COLLECTION
# --------------------------------------------------

print("=" * 60)
print("CHROMADB RETRIEVAL DEMO")
print("=" * 60)

print("Collection:", collection.name)
print("Total documents:", collection.count())


# --------------------------------------------------
# 1. COSINE SIMILARITY SEARCH
# --------------------------------------------------

query = "How does RAG retrieve information?"

print("\n" + "=" * 60)
print("1. COSINE SIMILARITY SEARCH")
print("=" * 60)

print("Query:", query)

results = collection.query(
    query_texts=[query],
    n_results=5
)


for i, document in enumerate(
    results["documents"][0],
    start=1
):

    print(f"\nResult {i}:")
    print(document)

    print("Metadata:")
    print(results["metadatas"][0][i - 1])


# --------------------------------------------------
# 2. METADATA FILTERING
# --------------------------------------------------

print("\n" + "=" * 60)
print("2. METADATA FILTERING")
print("=" * 60)

filtered_results = collection.query(
    query_texts=["information retrieval"],
    n_results=5,
    where={"category": "rag"}
)


for i, document in enumerate(
    filtered_results["documents"][0],
    start=1
):

    print(f"\nFiltered Result {i}:")
    print(document)

    print("Metadata:")
    print(filtered_results["metadatas"][0][i - 1])


# --------------------------------------------------
# 3. VERIFICATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

print("Similarity search completed successfully.")

print(
    "Metadata filtering completed successfully."
)

print(
    "All filtered results should have category = rag."
)