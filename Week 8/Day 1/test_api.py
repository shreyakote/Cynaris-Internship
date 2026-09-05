from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


# --------------------------------------------------
# Test root endpoint
# --------------------------------------------------


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# --------------------------------------------------
# Test health endpoint
# --------------------------------------------------


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# --------------------------------------------------
# Test prediction endpoint
# --------------------------------------------------


def test_predict():
    response = client.post(
        "/predict",
        json={
            "question": "What is RAG?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is RAG?"
    assert "answer" in data
    assert "latency_seconds" in data


# --------------------------------------------------
# Test empty question validation
# --------------------------------------------------


def test_empty_question():
    response = client.post(
        "/predict",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422
