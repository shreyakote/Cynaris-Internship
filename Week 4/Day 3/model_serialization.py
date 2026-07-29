"""
Week 4 Day 3
Model Serialization using Joblib & Pickle

AI/ML 3M Stack:
- MLflow
- MLOps
"""

import pickle
import joblib
import mlflow

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# MLflow
# -------------------------------------------------

mlflow.set_experiment("Week4_Day3_Model_Serialization")

with mlflow.start_run():

    # Train Model
    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")

    mlflow.log_metric("Accuracy", accuracy)

    # -------------------------------------------------
    # Save using Joblib
    # -------------------------------------------------

    joblib.dump(model, "logistic_model_joblib.pkl")

    # -------------------------------------------------
    # Save using Pickle
    # -------------------------------------------------

    with open("logistic_model_pickle.pkl", "wb") as file:
        pickle.dump(model, file)

    # Log Artifacts

    mlflow.log_artifact("logistic_model_joblib.pkl")
    mlflow.log_artifact("logistic_model_pickle.pkl")

# -------------------------------------------------
# Load Models
# -------------------------------------------------

joblib_model = joblib.load("logistic_model_joblib.pkl")

with open("logistic_model_pickle.pkl", "rb") as file:
    pickle_model = pickle.load(file)

# -------------------------------------------------
# Test Loaded Models
# -------------------------------------------------

joblib_predictions = joblib_model.predict(X_test)

pickle_predictions = pickle_model.predict(X_test)

print("\nTesting Saved Models")

print("Joblib Accuracy:",
      accuracy_score(y_test, joblib_predictions))

print("Pickle Accuracy:",
      accuracy_score(y_test, pickle_predictions))

# Verify predictions are identical

if (joblib_predictions == pickle_predictions).all():
    print("\nBoth serialized models produce identical predictions.")
else:
    print("\nPrediction mismatch detected.")

print("\nCompleted Successfully!")