from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


app = FastAPI(title="MLOps ML API")


# Load a small sample dataset
iris = load_iris()

X = iris.data
y = iris.target


# Train a simple ML model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42,
)

model.fit(X, y)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {
        "message": "MLOps ML API is running",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])

    return {
        "prediction": int(prediction[0]),
    }