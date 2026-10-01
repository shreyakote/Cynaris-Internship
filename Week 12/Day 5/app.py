from fastapi import FastAPI

app = FastAPI(title="3M Production AI System")


@app.get("/")
def root():
    return {
        "message": "3M Production AI System is running",
        "status": "success",
        "stack": [
            "CrewAI",
            "LangGraph",
            "MLflow",
            "Ragas",
            "MLOps",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}