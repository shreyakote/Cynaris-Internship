from pathlib import Path
import time

import chromadb
import mlflow

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ======================================================
# CONFIGURATION
# ======================================================

DOCUMENT_FOLDER = Path("documents")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "week7_rag"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 3


# ======================================================
# CHECK DOCUMENTS
# ======================================================

def check_documents():
    pdf_files = list(DOCUMENT_FOLDER.glob("*.pdf"))

    if len(pdf_files) == 0:
        raise FileNotFoundError(
            "No PDF documents found inside documents folder."
        )

    print(f"PDF files found: {len(pdf_files)}")

    for pdf in pdf_files:
        print("-", pdf.name)

    return pdf_files


# ======================================================
# CONFIGURE OLLAMA
# ======================================================

def configure_ollama():
    Settings.llm = Ollama(
        model=LLM_MODEL,
        request_timeout=120.0,
        context_window=4096,
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    )


# ======================================================
# LOAD DOCUMENTS
# ======================================================

def load_documents():
    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENT_FOLDER),
        required_exts=[".pdf"],
    ).load_data()

    return documents


# ======================================================
# CREATE INDEX
# ======================================================

def create_index(documents):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_or_create_collection(
        COLLECTION_NAME
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    start = time.perf_counter()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    indexing_time = time.perf_counter() - start

    return index, indexing_time


# ======================================================
# QUERY ENGINE
# ======================================================

def create_query_engine(index):
    return index.as_query_engine(similarity_top_k=TOP_K)


# ======================================================
# ASK QUESTION
# ======================================================

def ask(query_engine, question):
    start = time.perf_counter()

    response = query_engine.query(question)

    latency = time.perf_counter() - start

    sources = []

    for node in response.source_nodes:
        sources.append(
            node.metadata.get("file_name", "Unknown")
        )

    return str(response), sources, latency


# ======================================================
# MAIN
# ======================================================

def main():
    print("=" * 60)
    print("MULTI-DOCUMENT RAG SYSTEM")
    print("=" * 60)

    check_documents()

    configure_ollama()

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    index, indexing_time = create_index(documents)

    query_engine = create_query_engine(index)

    questions = [
        "What is Machine Learning?",
        "Explain Artificial Intelligence.",
        "What is Retrieval Augmented Generation?",
        "What is Cloud Computing?",
        "Explain Cybersecurity.",
    ]

    mlflow.set_experiment("Week7_Multi_Document_RAG")

    with mlflow.start_run():
        mlflow.log_param("llm", LLM_MODEL)
        mlflow.log_param("embedding", EMBEDDING_MODEL)
        mlflow.log_param("documents", len(documents))
        mlflow.log_metric("index_time", indexing_time)

        latencies = []

        for i, q in enumerate(questions, 1):
            print("\n" + "-" * 60)
            print(f"Q{i}: {q}")

            answer, sources, latency = ask(
                query_engine,
                q,
            )

            latencies.append(latency)

            print("\nAnswer:\n")
            print(answer)

            print("\nSources:")
            for s in set(sources):
                print("-", s)

            print(f"\nLatency: {latency:.2f}s")

        avg = sum(latencies) / len(latencies)

        mlflow.log_metric("average_latency", avg)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"Documents : {len(documents)}")
        print(f"Queries   : {len(questions)}")
        print(f"Avg Time  : {avg:.2f}s")

        print("\nRAG completed successfully!")


if __name__ == "__main__":
    main()