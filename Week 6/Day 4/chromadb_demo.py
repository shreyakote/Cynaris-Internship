import chromadb
from chromadb.utils import embedding_functions


# --------------------------------------------------
# CREATE CHROMADB CLIENT
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# --------------------------------------------------
# CREATE EMBEDDING FUNCTION
# --------------------------------------------------

embedding_function = embedding_functions.DefaultEmbeddingFunction()


# --------------------------------------------------
# CREATE COLLECTION
# --------------------------------------------------

collection = client.get_or_create_collection(
    name="rag_documents",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# 20 DOCUMENTS
# --------------------------------------------------

documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing focuses on human language.",
    "Computer vision allows computers to understand images.",
    "Artificial intelligence enables machines to perform intelligent tasks.",
    "RAG stands for Retrieval-Augmented Generation.",
    "RAG retrieves relevant information before generating an answer.",
    "ChromaDB is a vector database used for similarity search.",
    "Vector databases store numerical representations of data.",
    "Embeddings represent text as numerical vectors.",
    "Ollama allows users to run language models locally.",
    "LangChain is a framework for building LLM applications.",
    "Prompt engineering improves the quality of LLM responses.",
    "Agents can use tools to perform different tasks.",
    "Large language models can generate natural language.",
    "Cosine similarity measures similarity between vectors.",
    "Metadata can be used to filter vector database results.",
    "Document chunking divides large documents into smaller pieces.",
    "Top-k retrieval returns the most relevant documents.",
    "A RAG pipeline combines retrieval with generation."
]


# --------------------------------------------------
# METADATA
# --------------------------------------------------

metadatas = [
    {"category": "machine_learning"},
    {"category": "deep_learning"},
    {"category": "nlp"},
    {"category": "computer_vision"},
    {"category": "ai"},
    {"category": "rag"},
    {"category": "rag"},
    {"category": "chromadb"},
    {"category": "vector_database"},
    {"category": "embeddings"},
    {"category": "ollama"},
    {"category": "langchain"},
    {"category": "prompt_engineering"},
    {"category": "agents"},
    {"category": "llm"},
    {"category": "similarity"},
    {"category": "metadata"},
    {"category": "rag"},
    {"category": "retrieval"},
    {"category": "rag"}
]


# --------------------------------------------------
# ADD 20 DOCUMENTS
# --------------------------------------------------

ids = [f"doc_{i}" for i in range(1, 21)]

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)


# --------------------------------------------------
# VERIFY
# --------------------------------------------------

print("=" * 60)
print("CHROMADB COLLECTION CREATED")
print("=" * 60)

print("Collection name:", collection.name)
print("Number of documents:", collection.count())

print("\nDocuments added successfully.")