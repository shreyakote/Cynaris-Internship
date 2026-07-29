"""
Week 4 Day 4
FastAPI Model Serving
"""

import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
model = joblib.load("logistic_model.pkl")

# Create FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    version="1.0"
)

# Input Schema
class PatientData(BaseModel):

    features: list[float]

@app.get("/")
def home():

    return {
        "message": "FastAPI Model Serving is Running!"
    }

@app.post("/predict")
def predict(data: PatientData):

    prediction = model.predict([data.features])

    probability = model.predict_proba([data.features])

    return {
        "prediction": int(prediction[0]),
        "probability": probability.tolist()
    }