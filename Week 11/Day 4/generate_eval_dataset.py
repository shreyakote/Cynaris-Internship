import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag_pipeline import answer_question

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
OUTPUT_FILE = BASE_DIR / "ragas_dataset.json"

questions_and_answers = [
    {
        "question": "What is Artificial Intelligence?",
        "ground_truth": "Artificial Intelligence is the field of creating computer systems that perform tasks requiring human-like intelligence.",
    },
    {
        "question": "What is Machine Learning?",
        "ground_truth": "Machine Learning is a subset of AI that enables systems to learn patterns from data and make predictions without being explicitly programmed for every case.",
    },
    {
        "question": "What is Deep Learning?",
        "ground_truth": "Deep Learning is a subset of machine learning that uses multi-layer neural networks to learn complex patterns.",
    },
    {
        "question": "What does Natural Language Processing do?",
        "ground_truth": "Natural Language Processing enables computers to understand, process, and generate human language.",
    },
    {
        "question": "What is Computer Vision?",
        "ground_truth": "Computer Vision enables computers to analyze and understand images and videos.",
    },
    {
        "question": "What is Generative AI?",
        "ground_truth": "Generative AI creates new content such as text, images, audio, and code based on patterns learned from data.",
    },
    {
        "question": "What are Large Language Models?",
        "ground_truth": "Large Language Models are neural networks trained on large text datasets to understand and generate language.",
    },
    {
        "question": "What is Retrieval-Augmented Generation?",
        "ground_truth": "RAG combines information retrieval with language generation. It retrieves relevant documents and uses them to provide grounded answers.",
    },
    {
        "question": "What is ChromaDB used for?",
        "ground_truth": "ChromaDB is a vector database that stores embeddings and supports similarity search for retrieving relevant information.",
    },
    {
        "question": "What does MLOps involve?",
        "ground_truth": "MLOps applies machine learning practices to deployment, monitoring, automation, and maintenance of machine learning systems.",
    },
]


def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    vector_store = Chroma(
        collection_name="week11_rag",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    evaluation_rows = []

    for item in questions_and_answers:
        question = item["question"]
        answer, contexts = answer_question(vector_store, question)

        evaluation_rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")

    OUTPUT_FILE.write_text(
        json.dumps(evaluation_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved {len(evaluation_rows)} evaluation rows to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()