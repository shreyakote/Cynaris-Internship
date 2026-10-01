from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Production ML Monitoring API")

iris = load_iris()

X = iris.data
y = iris.target

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
        "message": "Production ML Monitoring API is running",
        "status": "success",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])

    return {
        "prediction": int(prediction[0]),
    }