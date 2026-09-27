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
RETRIEVAL_K = 3

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0)


def build_vector_store():
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    documents = splitter.create_documents([text])

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="week11_rag",
    )

    print(f"Indexed {len(documents)} document chunks.")
    return vector_store


def answer_question(vector_store, question):
    retrieved_docs = vector_store.similarity_search(
        question,
        k=RETRIEVAL_K,
    )

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""Answer the question using only the context below.
If the answer is not present in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:"""

    response = llm.invoke(prompt)

    return response.content, [doc.page_content for doc in retrieved_docs]


if __name__ == "__main__":
    store = build_vector_store()

    question = "What is Retrieval-Augmented Generation?"
    answer, contexts = answer_question(store, question)

    print("\nQuestion:", question)
    print("\nAnswer:", answer)
    print("\nRetrieved context:")
    for context in contexts:
        print("-", context)