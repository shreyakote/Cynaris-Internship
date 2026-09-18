import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_URI = "models:/Day2IrisRandomForest/1"

model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI()


class PredictionRequest(BaseModel):
    inputs: list[list[float]]


@app.get("/")
def home():
    return {"message": "Day 2 MLflow model API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    predictions = model.predict(request.inputs)

    return {
        "model": MODEL_URI,
        "predictions": predictions.tolist(),
    }
