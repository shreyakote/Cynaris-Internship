from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder
)


# ============================================================
# 1. Create Document Store
# ============================================================

document_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)

print("Document store created.")


# ============================================================
# 2. Load PDF documents
# ============================================================

pdf_folder = Path("documents")

converter = PyPDFToDocument()

all_documents = []

for pdf_file in pdf_folder.glob("*.pdf"):

    print(f"Loading: {pdf_file.name}")

    result = converter.run(
        sources=[pdf_file]
    )

    all_documents.extend(result["documents"])


print(f"\nTotal PDF documents loaded: {len(all_documents)}")


# ============================================================
# 3. Clean documents
# ============================================================

cleaner = DocumentCleaner()

clean_result = cleaner.run(
    documents=all_documents
)

documents = clean_result["documents"]

print(f"Documents after cleaning: {len(documents)}")


# ============================================================
# 4. Create Ollama Document Embedder
# ============================================================

document_embedder = OllamaDocumentEmbedder(
    model="nomic-embed-text",
    url="http://localhost:11434"
)


# ============================================================
# 5. Generate embeddings
# ============================================================

print("\nGenerating document embeddings...")

embedding_result = document_embedder.run(
    documents=documents
)

embedded_documents = embedding_result["documents"]

print("Document embeddings created.")


# ============================================================
# 6. Store embedded documents
# ============================================================

document_store.write_documents(
    embedded_documents
)

print("Embedded documents stored successfully.")


# ============================================================
# 7. Create Ollama Text Embedder
# ============================================================

text_embedder = OllamaTextEmbedder(
    model="nomic-embed-text",
    url="http://localhost:11434"
)


# ============================================================
# 8. Create Dense Retriever
# ============================================================

retriever = InMemoryEmbeddingRetriever(
    document_store=document_store,
    top_k=3
)


# ============================================================
# 9. Build Haystack Pipeline
# ============================================================

pipeline = Pipeline()

pipeline.add_component(
    "text_embedder",
    text_embedder
)

pipeline.add_component(
    "retriever",
    retriever
)

pipeline.connect(
    "text_embedder.embedding",
    "retriever.query_embedding"
)


# ============================================================
# 10. Ten Questions
# ============================================================

questions = [
    "What is machine learning?",
    "What is artificial intelligence?",
    "What is RAG?",
    "What is cloud computing?",
    "What is cybersecurity?",
    "What are the advantages of machine learning?",
    "What is the purpose of a vector database?",
    "What is retrieval augmented generation?",
    "What are common cybersecurity threats?",
    "What are the benefits of cloud computing?"
]


# ============================================================
# 11. Run Dense Retrieval
# ============================================================

print("\n")
print("=" * 60)
print("DENSE RETRIEVAL RESULTS")
print("=" * 60)

for i, question in enumerate(questions, start=1):

    result = pipeline.run(
        {
            "text_embedder": {
                "text": question
            }
        }
    )

    retrieved_documents = result["retriever"]["documents"]

    print(f"\nQuestion {i}: {question}")

    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        print(f"\nRank {rank}")
        print(f"Score: {document.score}")

        print(
            "Content:", 
            document.content[:300].replace("\n", " ")
        )

        if document.meta:
            print("Metadata:", document.meta)