"""
Week 4 Day 5
1M Capstone: Sentiment Classifier – Deploy & Document

AI/ML 3M Stack:
- CrewAI (workflow reference)
- LangGraph (pipeline planning)
- MLflow (experiment tracking)
- MLOps (versioning & reproducibility)
"""

import os
import mlflow
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import movie_reviews

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# ----------------------------------------------------
# Download dataset (first run only)
# ----------------------------------------------------
nltk.download("movie_reviews")

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        review = movie_reviews.raw(fileid)
        documents.append((review, category))

texts = [text for text, label in documents]
labels = [1 if label == "pos" else 0 for text, label in documents]

# ----------------------------------------------------
# Train-Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ----------------------------------------------------
# Models
# ----------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

# ----------------------------------------------------
# MLflow
# ----------------------------------------------------
mlflow.set_experiment("Week4_Day5_Sentiment_Classifier")

results = {}

with mlflow.start_run():

    plt.figure(figsize=(8,6))

    for name, model in models.items():

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "AUC": auc
        }

        print("\n" + "="*60)
        print(name)
        print("="*60)

        print(classification_report(y_test, predictions))

        mlflow.log_metric(f"{name}_Accuracy", accuracy)
        mlflow.log_metric(f"{name}_Precision", precision)
        mlflow.log_metric(f"{name}_Recall", recall)
        mlflow.log_metric(f"{name}_ROC_AUC", auc)

        # --------------------------
        # Confusion Matrix
        # --------------------------

        cm = confusion_matrix(y_test, predictions)

        disp = ConfusionMatrixDisplay(cm)

        disp.plot()

        filename = (
            f"confusion_matrix_"
            f"{name.lower().replace(' ','_')}.png"
        )

        plt.savefig(filename)
        plt.close()

        mlflow.log_artifact(filename)

        # --------------------------
        # ROC Curve
        # --------------------------

        fpr, tpr, _ = roc_curve(y_test, probabilities)

        plt.figure(1)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC={auc:.3f})"
        )

# ROC Plot

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig("roc_curve_comparison.png")
plt.close()

mlflow.log_artifact("roc_curve_comparison.png")

# ----------------------------------------------------
# Results Table
# ----------------------------------------------------
print("\n")
print("="*65)
print("MODEL COMPARISON")
print("="*65)

for model, metric in results.items():

    print(f"\n{model}")

    print(f"Accuracy : {metric['Accuracy']:.4f}")
    print(f"Precision: {metric['Precision']:.4f}")
    print(f"Recall   : {metric['Recall']:.4f}")
    print(f"ROC-AUC  : {metric['AUC']:.4f}")

print("\nCompleted Successfully!")

print("\nGenerated Files:")

for file in os.listdir():
    if file.endswith(".png"):
        print(file)