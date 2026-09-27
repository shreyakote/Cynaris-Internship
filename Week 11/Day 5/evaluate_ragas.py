
"""Run a lightweight evaluation of the RAG pipeline."""

from pathlib import Path

import pandas as pd

from rag_pipeline import answer_question


BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "ragas_results.csv"

QUESTIONS = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is deep learning?",
    "What is natural language processing?",
    "What is computer vision?",
    "What is generative AI?",
    "What are large language models?",
    "What is retrieval-augmented generation?",
    "What is ChromaDB?",
    "What is MLOps?",
]


def main():
    """Evaluate whether the RAG pipeline returns answers and contexts."""

    rows = []

    print("Running quick RAG evaluation...")

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"Question {index}/{len(QUESTIONS)}: {question}")

        result = answer_question(question)

        answer = str(result.get("answer", "")).strip()
        contexts = result.get("contexts", [])

        # Basic checks only; these are not Ragas metric scores.
        answer_present = bool(answer)
        context_present = bool(contexts)
        answer_length = len(answer)

        rows.append(
            {
                "question": question,
                "answer": answer,
                "context_count": len(contexts),
                "answer_present": answer_present,
                "context_present": context_present,
                "answer_length": answer_length,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_FILE, index=False)

    print("\nQuick evaluation completed.")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Questions evaluated: {len(results_df)}")
    print(
        "Answers returned:",
        int(results_df["answer_present"].sum()),
        "/",
        len(results_df),
    )
    print(
        "Questions with retrieved context:",
        int(results_df["context_present"].sum()),
        "/",
        len(results_df),
    )

    print("\nDetailed results:")
    print(results_df)


if __name__ == "__main__":
    main()