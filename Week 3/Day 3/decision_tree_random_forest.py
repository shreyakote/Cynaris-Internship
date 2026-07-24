# ============================================
# Week 3 - Day 3
# Decision Trees & Random Forests
# AI/ML Internship Task
# ============================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================
# Load Dataset
# ============================================

print("Loading Breast Cancer Dataset...")

data = load_breast_cancer()

X = data.data
y = data.target

# ============================================
# Split Dataset
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# Decision Tree Model
# ============================================

print("\nTraining Decision Tree...")

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

# ============================================
# Decision Tree Evaluation
# ============================================

print("\n========== Decision Tree ==========")

dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f"Accuracy : {dt_accuracy:.4f}")

print("\nClassification Report")

print(classification_report(y_test, dt_predictions))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, dt_predictions))

# Save confusion matrix image

ConfusionMatrixDisplay.from_predictions(y_test, dt_predictions)

plt.title("Decision Tree Confusion Matrix")

plt.savefig("confusion_matrix_dt.png")

plt.close()

# ============================================
# Random Forest Model
# ============================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# ============================================
# Random Forest Evaluation
# ============================================

print("\n========== Random Forest ==========")

rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Accuracy : {rf_accuracy:.4f}")

print("\nClassification Report")

print(classification_report(y_test, rf_predictions))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, rf_predictions))

ConfusionMatrixDisplay.from_predictions(y_test, rf_predictions)

plt.title("Random Forest Confusion Matrix")

plt.savefig("confusion_matrix_rf.png")

plt.close()

# ============================================
# Feature Importance
# ============================================

print("\nCreating Feature Importance Plot...")

importance = rf_model.feature_importances_

feature_names = data.feature_names

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(12,6))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xticks(rotation=90)

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.close()

print("\nTop 10 Important Features")

print(importance_df.head(10))

# ============================================
# MLflow Logging
# ============================================

print("\nLogging Experiment using MLflow...")

mlflow.set_experiment("Week3_DecisionTree_RandomForest")

with mlflow.start_run():

    mlflow.log_param("Model", "RandomForest")

    mlflow.log_param("Trees", 100)

    mlflow.log_metric("Accuracy", rf_accuracy)

    mlflow.sklearn.log_model(
        rf_model,
        "RandomForestModel"
    )

# ============================================
# Save Model
# ============================================

joblib.dump(
    rf_model,
    "random_forest_model.pkl"
)

print("\nModel Saved Successfully!")

print("\nTask Completed Successfully!")