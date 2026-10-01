from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert "CrewAI" in data["stack"]
    assert "LangGraph" in data["stack"]
    assert "MLflow" in data["stack"]
    assert "Ragas" in data["stack"]
    assert "MLOps" in data["stack"]


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"