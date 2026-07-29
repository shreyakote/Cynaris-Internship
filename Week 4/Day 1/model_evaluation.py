"""
Week 4 Day 1
Model Evaluation Metrics
Precision, Recall, F1 Score, ROC-AUC

AI/ML 3M Stack:
- MLflow
- MLOps
"""

import mlflow
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# Load Dataset
# -----------------------------
data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MLflow Tracking
# -----------------------------
mlflow.set_experiment("Week4_Day1_Model_Evaluation")

with mlflow.start_run():

    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Metrics

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC AUC  :", auc)

    # Log Metrics

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1", f1)
    mlflow.log_metric("ROC_AUC", auc)

    # -----------------------------
    # ROC Curve
    # -----------------------------

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

    plt.close()

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    plt.close()

print("\nCompleted Successfully!")