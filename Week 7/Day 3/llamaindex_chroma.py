from pathlib import Path
import time

import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FOLDER = Path("documents")

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "llamaindex_documents"

LLM_MODEL = "llama3.2:3b"

EMBEDDING_MODEL = "nomic-embed-text:latest"


# ============================================================
# CHECK DOCUMENTS
# ============================================================

DOCUMENT_FOLDER.mkdir(exist_ok=True)

txt_files = list(DOCUMENT_FOLDER.glob("*.txt"))

print("=" * 60)
print("LLAMAINDEX + CHROMADB RAG")
print("=" * 60)

print(f"Text files found: {len(txt_files)}")

if len(txt_files) == 0:
    print("ERROR: No .txt files found!")
    exit()


# ============================================================
# CONFIGURE OLLAMA
# ============================================================

Settings.llm = Ollama(
    model=LLM_MODEL,
    request_timeout=120.0,
    context_window=4096
)

Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)

print("\nLLM:", LLM_MODEL)
print("Embedding:", EMBEDDING_MODEL)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

print("\nLoading documents...")

documents = SimpleDirectoryReader(
    input_dir=str(DOCUMENT_FOLDER),
    required_exts=[".txt"]
).load_data()

print(f"Documents loaded: {len(documents)}")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("\nConnecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

chroma_collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# ============================================================
# CREATE VECTOR INDEX
# ============================================================

print("\nCreating ChromaDB VectorStoreIndex...")
print("Generating Ollama embeddings...")

start_indexing = time.perf_counter()

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

indexing_time = time.perf_counter() - start_indexing

print(f"\nIndexing completed in {indexing_time:.2f} seconds")


# ============================================================
# CREATE QUERY ENGINE
# ============================================================

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("\nQueryEngine created successfully!")


# ============================================================
# QUESTIONS
# ============================================================

questions = [

    "What is artificial intelligence?",

    "What are the applications of artificial intelligence?",

    "What is machine learning?",

    "What are the types of machine learning?",

    "What is deep learning?",

    "What are the applications of deep learning?",

    "What is natural language processing?",

    "What are the applications of NLP?",

    "What is computer vision?",

    "What are the applications of computer vision?"
]


# ============================================================
# RUN QUERIES
# ============================================================

print("\n" + "=" * 60)
print("RUNNING 10 CHROMADB QUERIES")
print("=" * 60)

latencies = []

for number, question in enumerate(questions, start=1):

    print("\n" + "-" * 60)

    print(f"Q{number}. {question}")

    start_time = time.perf_counter()

    response = query_engine.query(question)

    latency = time.perf_counter() - start_time

    latencies.append(latency)

    print("\nAnswer:")
    print(response)

    print(f"\nLatency: {latency:.2f} seconds")

    print("\nSource documents:")

    sources = set()

    for node in response.source_nodes:

        file_name = node.metadata.get(
            "file_name",
            "Unknown"
        )

        sources.add(file_name)

    for source in sources:

        print("-", source)


# ============================================================
# AVERAGE LATENCY
# ============================================================

average_latency = sum(latencies) / len(latencies)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("CHROMADB SUMMARY")
print("=" * 60)

print(f"Documents indexed : {len(documents)}")

print(f"Queries executed  : {len(questions)}")

print(f"Average latency   : {average_latency:.2f} seconds")

print(f"Indexing time     : {indexing_time:.2f} seconds")

print("\nSUCCESS!")

print("LlamaIndex + ChromaDB RAG completed successfully.")