from pathlib import Path

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


# --------------------------------------------------
# 1. Create document store
# --------------------------------------------------

document_store = InMemoryDocumentStore()


# --------------------------------------------------
# 2. Load PDF documents
# --------------------------------------------------

pdf_folder = Path("documents")

pdf_converter = PyPDFToDocument()

all_documents = []

for pdf_file in pdf_folder.glob("*.pdf"):
    print(f"Loading: {pdf_file.name}")

    result = pdf_converter.run(
        sources=[pdf_file]
    )

    documents = result["documents"]

    all_documents.extend(documents)


print("\nTotal PDF documents loaded:", len(all_documents))


# --------------------------------------------------
# 3. Clean documents
# --------------------------------------------------

cleaner = DocumentCleaner()

clean_result = cleaner.run(
    documents=all_documents
)

clean_documents = clean_result["documents"]

print("Total documents after cleaning:", len(clean_documents))


# --------------------------------------------------
# 4. Write documents into DocumentStore
# --------------------------------------------------

document_store.write_documents(clean_documents)

print("Documents stored successfully!")


# --------------------------------------------------
# 5. Create BM25 Retriever
# --------------------------------------------------

retriever = InMemoryBM25Retriever(
    document_store=document_store,
    top_k=3
)


# --------------------------------------------------
# 6. Create Haystack Pipeline
# --------------------------------------------------

pipeline = Pipeline()

pipeline.add_component(
    "retriever",
    retriever
)


# --------------------------------------------------
# 7. Ask questions
# --------------------------------------------------

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


# --------------------------------------------------
# 8. Run 10 questions
# --------------------------------------------------

print("\n========== BM25 RETRIEVAL RESULTS ==========\n")

for i, question in enumerate(questions, start=1):

    result = pipeline.run(
        {
            "retriever": {
                "query": question
            }
        }
    )

    documents = result["retriever"]["documents"]

    print(f"\nQuestion {i}: {question}")

    for rank, document in enumerate(documents, start=1):

        print(f"\nRank {rank}")
        print("Score:", document.score)
        print("Content:", document.content[:300])