import chromadb

# Create persistent ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create collection
collection = client.get_or_create_collection(
    name="practical_documents",
    metadata={"hnsw:space": "cosine"}
)

# 20 documents
documents = [
    "Python is a popular programming language used for data science.",
    "Machine learning allows computers to learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Artificial intelligence enables machines to perform intelligent tasks.",
    "Cloud computing provides computing resources over the internet.",
    "Docker is used to package applications into containers.",
    "Kubernetes manages containerized applications.",
    "Git is a distributed version control system.",
    "GitHub provides hosting for Git repositories.",
    "FastAPI is a Python framework for building APIs.",
    "LangChain is used to build applications powered by language models.",
    "ChromaDB is a vector database designed for AI applications.",
    "RAG combines retrieval with language model generation.",
    "Ollama allows users to run large language models locally.",
    "Embeddings represent text as numerical vectors.",
    "Cosine similarity measures similarity between vectors.",
    "Metadata can be used to filter vector database results.",
    "PDF documents can be split into smaller chunks.",
    "Vector databases are useful for semantic search.",
    "Large language models can answer questions using retrieved context."
]

ids = [f"doc_{i}" for i in range(1, 21)]

metadatas = [
    {"category": "programming"},
    {"category": "machine-learning"},
    {"category": "deep-learning"},
    {"category": "ai"},
    {"category": "cloud"},
    {"category": "devops"},
    {"category": "devops"},
    {"category": "git"},
    {"category": "git"},
    {"category": "api"},
    {"category": "llm"},
    {"category": "vector-db"},
    {"category": "rag"},
    {"category": "llm"},
    {"category": "embeddings"},
    {"category": "vector-db"},
    {"category": "vector-db"},
    {"category": "rag"},
    {"category": "vector-db"},
    {"category": "llm"},
]

# Add documents
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("\n===== CHROMADB SETUP =====")
print("Documents added successfully!")
print("Total documents:", collection.count())


# Similarity search
query = "How can vector databases be used for searching information?"

results = collection.query(
    query_texts=[query],
    n_results=5
)

print("\n===== COSINE SIMILARITY SEARCH =====")
print("Query:", query)

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}")
    print("ID:", results["ids"][0][i])
    print("Document:", document)
    print("Metadata:", results["metadatas"][0][i])
    print("Distance:", results["distances"][0][i])


# Metadata filtering
print("\n===== METADATA FILTERING =====")

filtered_results = collection.get(
    where={"category": "vector-db"}
)

for i, document in enumerate(filtered_results["documents"]):
    print(f"\nDocument {i + 1}")
    print("Document:", document)
    print("Metadata:", filtered_results["metadatas"][i])