"""
Week 4 Day 4
Train Logistic Regression Model
"""

import joblib
import mlflow

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlflow.set_experiment("Week4_Day4_FastAPI")

with mlflow.start_run():

    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    mlflow.log_metric("Accuracy", accuracy)

    joblib.dump(model, "logistic_model.pkl")

    mlflow.log_artifact("logistic_model.pkl")

print("Model Saved Successfully")