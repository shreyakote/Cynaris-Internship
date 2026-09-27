
from pathlib import Path

import mlflow
import pandas as pd

from rag_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RETRIEVER_K,
    answer_question,
)

BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "ragas_results.csv"

mlflow.set_tracking_uri((BASE_DIR / "mlruns").as_uri())
mlflow.set_experiment("Week 11 Tracked RAG Pipeline")


def track_rag_run(question: str):
    """Run RAG and record its configuration and answer in MLflow."""
    result = answer_question(question)

    with mlflow.start_run():
        mlflow.log_param("llm_model", LLM_MODEL)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_param("retriever_k", RETRIEVER_K)

        mlflow.log_text(question, "question.txt")
        mlflow.log_text(result["answer"], "answer.txt")
        mlflow.log_text(
            "\n\n".join(result["contexts"]),
            "retrieved_contexts.txt",
        )

        print("RAG run tracked in MLflow.")
        print("Answer:", result["answer"])

    return result


def log_ragas_results():
    """Log available Ragas scores to MLflow."""
    if not RESULTS_FILE.exists():
        print("No Ragas results found. Run evaluate_ragas.py first.")
        return

    dataframe = pd.read_csv(RESULTS_FILE)

    score_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    with mlflow.start_run(run_name="Ragas Evaluation"):
        for column in score_columns:
            if column in dataframe.columns:
                score = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                ).mean()

                if pd.notna(score):
                    mlflow.log_metric(f"mean_{column}", float(score))

        mlflow.log_param("evaluation_examples", len(dataframe))
        mlflow.log_artifact(str(RESULTS_FILE))

    print("Ragas metrics and CSV logged to MLflow.")


if __name__ == "__main__":
    track_rag_run(
        "What is Retrieval-Augmented Generation?"
    )
    log_ragas_results()