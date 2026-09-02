from pathlib import Path

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


# --------------------------------------------------
# 1. Create Document Store
# --------------------------------------------------

document_store = InMemoryDocumentStore()


# --------------------------------------------------
# 2. Load 5 PDF documents
# --------------------------------------------------

data_folder = Path("data")
pdf_files = list(data_folder.glob("*.pdf"))

print("PDF files found:", len(pdf_files))

documents = []

for pdf_file in pdf_files:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_file))

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if text.strip():
            documents.append(
                Document(
                    content=text,
                    meta={"file_name": pdf_file.name}
                )
            )

            print("Loaded:", pdf_file.name)

    except Exception as e:
        print("Error reading", pdf_file.name, ":", e)


# --------------------------------------------------
# 3. Write documents to Document Store
# --------------------------------------------------

document_store.write_documents(documents)

print("\nTotal documents indexed:", len(documents))


# --------------------------------------------------
# 4. Create BM25 Retriever
# --------------------------------------------------

retriever = InMemoryBM25Retriever(
    document_store=document_store
)


# --------------------------------------------------
# 5. Create Haystack Pipeline
# --------------------------------------------------

pipeline = Pipeline()

pipeline.add_component(
    "retriever",
    retriever
)


# --------------------------------------------------
# 6. Ask questions
# --------------------------------------------------

questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is deep learning?",
    "What is natural language processing?",
    "What is computer vision?",
    "What are the applications of artificial intelligence?",
    "What are the advantages of machine learning?",
    "What is data preprocessing?",
    "What is supervised learning?",
    "What is unsupervised learning?"
]


# --------------------------------------------------
# 7. Run retrieval
# --------------------------------------------------

for i, question in enumerate(questions, start=1):

    print("\n" + "=" * 70)
    print(f"Question {i}: {question}")

    result = pipeline.run(
        {
            "retriever": {
                "query": question,
                "top_k": 3
            }
        }
    )

    retrieved_docs = result["retriever"]["documents"]

    for rank, doc in enumerate(retrieved_docs, start=1):

        print(f"\nRank {rank}")
        print("File:", doc.meta.get("file_name"))
        print("Score:", doc.score)

        preview = doc.content[:300].replace("\n", " ")

        print("Content:", preview)