from fastapi import FastAPI
from pydantic import BaseModel, Field
import time
import mlflow


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Production RAG ML API",
    description="Dockerised ML/RAG API for MLOps",
    version="1.0.0",
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Production RAG ML API is running",
        "status": "healthy"
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction / RAG endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: QueryRequest):

    start_time = time.perf_counter()

    question = request.question

    # --------------------------------------------------
    # Demonstration RAG response
    # --------------------------------------------------
    # Replace this section later with the actual
    # ChromaDB + LLM RAG pipeline.

    answer = (
        f"RAG system received the question: {question}"
    )

    latency = time.perf_counter() - start_time

    # --------------------------------------------------
    # MLflow tracking
    # --------------------------------------------------

    try:
        mlflow.set_experiment("Production_RAG_API")

        with mlflow.start_run():

            mlflow.log_param(
                "question_length",
                len(question)
            )

            mlflow.log_metric(
                "latency_seconds",
                latency
            )

    except Exception:
        # The API continues working if MLflow
        # tracking is temporarily unavailable.
        pass

    # --------------------------------------------------
    # API response
    # --------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "latency_seconds": round(latency, 4)
    }