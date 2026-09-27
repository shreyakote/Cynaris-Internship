
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge_base.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 3


def build_vector_store():
    """Load the knowledge base, split it, and create a Chroma vector store."""
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_FILE}"
        )

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([text])

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="week11_tracked_rag",
    )

    return vector_store


def answer_question(question: str):
    """Retrieve relevant context and generate an answer."""
    vector_store = build_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )
    documents = retriever.invoke(question)

    contexts = [document.page_content for document in documents]
    context_text = "\n\n".join(contexts)

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = f"""
Answer the question using only the context below.
If the context does not contain the answer, say that you do not know.
Do not invent facts.

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "contexts": contexts,
    }


if __name__ == "__main__":
    question = "What is Retrieval-Augmented Generation?"
    result = answer_question(question)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved contexts:")
    for index, context in enumerate(result["contexts"], start=1):
        print(f"\nContext {index}:")
        print(context)