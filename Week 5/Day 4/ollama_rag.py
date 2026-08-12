import os

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# ============================================
# 1. LOAD PDF
# ============================================

PDF_PATH = "sample.pdf"

if not os.path.exists(PDF_PATH):
    print("ERROR: sample.pdf not found.")
    exit()

print("\nLoading PDF...")

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print("PDF loaded successfully!")
print("Number of pages:", len(pages))


# ============================================
# 2. SPLIT PDF INTO CHUNKS
# ============================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(pages)

print("Number of chunks:", len(chunks))

if len(chunks) == 0:
    print("ERROR: No chunks were created.")
    exit()


# ============================================
# 3. CREATE EMBEDDINGS
# ============================================

print("\nCreating embeddings...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ============================================
# 4. CREATE CHROMADB
# ============================================

client = chromadb.PersistentClient(
    path="./pdf_chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)


# ============================================
# 5. STORE PDF CHUNKS
# ============================================

print("\nStoring PDF chunks in ChromaDB...")

for i, chunk in enumerate(chunks):

    text = chunk.page_content

    embedding = embeddings.embed_query(text)

    collection.upsert(
        ids=[f"chunk_{i}"],
        documents=[text],
        embeddings=[embedding],
        metadatas=[
            {
                "source": PDF_PATH,
                "page": chunk.metadata.get("page", 0)
            }
        ]
    )

print("Embeddings stored successfully!")
print("Total chunks in ChromaDB:", collection.count())


# ============================================
# 6. ASK QUESTION
# ============================================

question = input("\nEnter your question: ")


# ============================================
# 7. CREATE QUESTION EMBEDDING
# ============================================

query_embedding = embeddings.embed_query(question)


# ============================================
# 8. RETRIEVE TOP 3 CHUNKS
# ============================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


print("\n===== TOP 3 RETRIEVED CHUNKS =====")

retrieved_chunks = []

for i, document in enumerate(results["documents"][0]):

    print(f"\n--- Chunk {i + 1} ---")

    print("Distance:", results["distances"][0][i])

    print("Metadata:", results["metadatas"][0][i])

    print("Text:")
    print(document)

    retrieved_chunks.append(document)


# ============================================
# 9. COMBINE CONTEXT
# ============================================

context = "\n\n".join(retrieved_chunks)


# ============================================
# 10. CONNECT TO OLLAMA
# ============================================

print("\nConnecting to Ollama...")

llm = OllamaLLM(
    model="llama3.2:3b"
)


# ============================================
# 11. CREATE PROMPT
# ============================================

prompt = f"""
You are a helpful assistant.

Use ONLY the information in the context below
to answer the question.

Do not use outside knowledge.

If the answer is not present in the context,
say:

"I cannot find the answer in the provided PDF."

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""


# ============================================
# 12. GENERATE ANSWER
# ============================================

print("\n===== OLLAMA ANSWER =====")

try:

    answer = llm.invoke(prompt)

    print(answer)

except Exception as error:

    print("Ollama error occurred:")
    print(error)