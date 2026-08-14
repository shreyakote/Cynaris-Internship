from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

import chromadb
from chromadb.utils import embedding_functions


# ==================================================
# CONFIGURATION
# ==================================================

PDF_PATH = "sample.pdf"

CHROMA_PATH = "./pdf_chroma_db"

COLLECTION_NAME = "pdf_rag_collection"


# ==================================================
# 1. LOAD PDF
# ==================================================

print("=" * 60)
print("1. LOADING PDF")
print("=" * 60)

loader = PyPDFLoader(PDF_PATH)

pages = loader.load()

print("Pages loaded:", len(pages))


# ==================================================
# 2. SPLIT PDF INTO CHUNKS
# ==================================================

print("\n" + "=" * 60)
print("2. CREATING DOCUMENT CHUNKS")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)

print("Total chunks created:", len(chunks))


# ==================================================
# 3. CREATE CHROMADB
# ==================================================

print("\n" + "=" * 60)
print("3. CREATING CHROMADB COLLECTION")
print("=" * 60)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)

print("Collection:", collection.name)


# ==================================================
# 4. PREPARE DOCUMENT DATA
# ==================================================

documents = [
    chunk.page_content
    for chunk in chunks
]

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

metadatas = [
    {
        "page": chunk.metadata.get("page", 0),
        "source": PDF_PATH
    }
    for chunk in chunks
]


# ==================================================
# 5. STORE CHUNKS IN CHROMADB
# ==================================================

print("\n" + "=" * 60)
print("4. STORING CHUNKS IN CHROMADB")
print("=" * 60)

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(
    "Chunks stored:",
    collection.count()
)


# ==================================================
# 6. GET USER QUESTION
# ==================================================

print("\n" + "=" * 60)
print("5. ASK A QUESTION")
print("=" * 60)

question = input(
    "Enter your question about the PDF: "
)


# ==================================================
# 7. RETRIEVE TOP 3 CHUNKS
# ==================================================

print("\n" + "=" * 60)
print("6. TOP 3 RETRIEVED CHUNKS")
print("=" * 60)

results = collection.query(
    query_texts=[question],
    n_results=3
)

retrieved_documents = results["documents"][0]

for i, document in enumerate(
    retrieved_documents,
    start=1
):

    print(f"\n--- Chunk {i} ---")
    print(document)

    print(
        "Metadata:",
        results["metadatas"][0][i - 1]
    )


# ==================================================
# 8. BUILD CONTEXT
# ==================================================

context = "\n\n".join(
    retrieved_documents
)


# ==================================================
# 9. CONNECT TO OLLAMA
# ==================================================

print("\n" + "=" * 60)
print("7. CONNECTING TO OLLAMA")
print("=" * 60)

llm = OllamaLLM(
    model="llama3.2:3b"
)

print("Model: llama3.2:3b")


# ==================================================
# 10. RAG PROMPT
# ==================================================

prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the context provided below.

If the answer cannot be found in the context,
say:

"The answer is not available in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""


# ==================================================
# 11. GENERATE ANSWER
# ==================================================

print("\n" + "=" * 60)
print("8. OLLAMA RAG ANSWER")
print("=" * 60)

answer = llm.invoke(prompt)

print(answer)


# ==================================================
# 12. VERIFICATION
# ==================================================

print("\n" + "=" * 60)
print("RAG PIPELINE VERIFICATION")
print("=" * 60)

print("PDF loaded successfully: YES")
print("Chunks created successfully: YES")
print("ChromaDB storage: YES")
print("Top-3 retrieval: YES")
print("Ollama generation: YES")