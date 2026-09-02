from pathlib import Path

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)

from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever
)

from pypdf import PdfReader


# ============================================================
# DAY 2 - HAYSTACK DENSE RETRIEVAL
# ============================================================

print("=" * 70)
print("HAYSTACK DAY 2 - DENSE RETRIEVAL")
print("=" * 70)


# ============================================================
# 1. DOCUMENTS FOLDER
# ============================================================

data_folder = Path("documents")

if not data_folder.exists():
    print("\nERROR: 'documents' folder not found!")
    print("Please make sure your PDF folder is named 'documents'.")
    print("Expected location:")
    print(data_folder.resolve())
    exit()


# ============================================================
# 2. FIND PDF FILES
# ============================================================

pdf_files = list(data_folder.glob("*.pdf"))

print(f"\nPDF files found: {len(pdf_files)}")


if len(pdf_files) == 0:
    print("\nERROR: No PDF files found!")
    print("Please put your PDF files inside:")
    print(data_folder.resolve())
    exit()


# ============================================================
# 3. LOAD PDF DOCUMENTS
# ============================================================

documents = []

print("\nLoading PDF documents...")
print("-" * 70)


for pdf_file in pdf_files:

    try:

        reader = PdfReader(str(pdf_file))

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"


        if text.strip():

            document = Document(
                content=text,
                meta={
                    "file_name": pdf_file.name
                }
            )

            documents.append(document)

            print(f"✓ Loaded: {pdf_file.name}")
            print(f"  Pages: {len(reader.pages)}")
            print(f"  Characters: {len(text)}")

        else:

            print(
                f"⚠ WARNING: No text extracted from "
                f"{pdf_file.name}"
            )


    except Exception as e:

        print(f"✗ Error reading {pdf_file.name}")
        print(f"  Error: {e}")


# ============================================================
# 4. DISPLAY TOTAL DOCUMENTS
# ============================================================

print("\n" + "=" * 70)

print(
    f"TOTAL DOCUMENTS LOADED: {len(documents)}"
)

print("=" * 70)


if len(documents) == 0:

    print("\nERROR: No readable PDF documents found.")
    exit()


# ============================================================
# 5. CREATE DOCUMENT STORE
# ============================================================

print("\nCreating InMemoryDocumentStore...")

document_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)

print("✓ Document store created.")


# ============================================================
# 6. CREATE DOCUMENT EMBEDDER
# ============================================================

print("\nLoading Sentence Transformer model...")

print(
    "Model: sentence-transformers/all-MiniLM-L6-v2"
)

document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

document_embedder.warm_up()

print("✓ Sentence Transformer model loaded.")


# ============================================================
# 7. GENERATE DOCUMENT EMBEDDINGS
# ============================================================

print("\nGenerating embeddings for documents...")

embedding_result = document_embedder.run(
    documents
)

embedded_documents = embedding_result["documents"]

print(
    f"✓ Generated embeddings for "
    f"{len(embedded_documents)} documents."
)


# ============================================================
# 8. WRITE DOCUMENTS TO DOCUMENT STORE
# ============================================================

print("\nWriting documents to Document Store...")

document_store.write_documents(
    embedded_documents
)

print("✓ Documents stored successfully.")


# ============================================================
# 9. CREATE TEXT EMBEDDER
# ============================================================

print("\nCreating query text embedder...")

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text_embedder.warm_up()

print("✓ Query text embedder ready.")


# ============================================================
# 10. CREATE DENSE RETRIEVER
# ============================================================

print("\nCreating Dense Retriever...")

retriever = InMemoryEmbeddingRetriever(
    document_store=document_store
)

print("✓ Dense Retriever created.")


# ============================================================
# 11. CREATE HAYSTACK PIPELINE
# ============================================================

print("\nCreating Haystack Dense Retrieval Pipeline...")

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


print("✓ Pipeline created successfully.")


# ============================================================
# 12. TEN TEST QUESTIONS
# ============================================================

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


# ============================================================
# 13. RUN DENSE RETRIEVAL
# ============================================================

print("\n")
print("=" * 70)
print("DENSE RETRIEVAL RESULTS")
print("=" * 70)


for question_number, question in enumerate(
    questions,
    start=1
):

    print("\n")
    print("-" * 70)

    print(
        f"QUESTION {question_number}: "
        f"{question}"
    )

    print("-" * 70)


    try:

        result = pipeline.run(
            {
                "text_embedder": {
                    "text": question
                },

                "retriever": {
                    "top_k": 3
                }
            }
        )


        retrieved_documents = result[
            "retriever"
        ]["documents"]


        if not retrieved_documents:

            print("No documents retrieved.")

            continue


        # ====================================================
        # DISPLAY TOP 3 DOCUMENTS
        # ====================================================

        for rank, document in enumerate(
            retrieved_documents,
            start=1
        ):

            file_name = document.meta.get(
                "file_name",
                "Unknown"
            )


            score = document.score


            content = document.content


            preview = content[:500]


            preview = preview.replace(
                "\n",
                " "
            )


            print(
                f"\nRank {rank}"
            )

            print(
                f"File   : {file_name}"
            )

            print(
                f"Score  : {score:.4f}"
            )

            print(
                f"Content: {preview}..."
            )


    except Exception as e:

        print(
            f"\nERROR while processing "
            f"Question {question_number}:"
        )

        print(e)


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("DENSE RETRIEVAL COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nTotal questions processed: 10")
print("Top documents retrieved per question: 3")

print("\nNext steps:")
print("1. Check the retrieved documents.")
print("2. Mark relevant / not relevant.")
print("3. Calculate Precision@3.")
print("4. Compare with BM25 retrieval.")
print("5. Record the results in comparison.txt.")

print("\nDay 2 Dense Retrieval task completed!")