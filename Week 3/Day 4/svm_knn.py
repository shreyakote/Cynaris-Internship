# ============================================
# Week 3 - Day 4
# SVM & KNN Classification
# ============================================

import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================
# Load Dataset
# ============================================

print("Loading Dataset...")

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
# Feature Scaling
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================
# Support Vector Machine
# ============================================

print("\nTraining SVM...")

svm = SVC(kernel="rbf", random_state=42)

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n========== SVM ==========")
print("Accuracy:", svm_accuracy)
print(classification_report(y_test, svm_pred))

ConfusionMatrixDisplay.from_predictions(y_test, svm_pred)
plt.title("SVM Confusion Matrix")
plt.savefig("svm_confusion_matrix.png")
plt.close()

# ============================================
# K-Nearest Neighbors
# ============================================

print("\nTraining KNN...")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("\n========== KNN ==========")
print("Accuracy:", knn_accuracy)
print(classification_report(y_test, knn_pred))

ConfusionMatrixDisplay.from_predictions(y_test, knn_pred)
plt.title("KNN Confusion Matrix")
plt.savefig("knn_confusion_matrix.png")
plt.close()

# ============================================
# Compare Results
# ============================================

results = pd.DataFrame({
    "Model": ["SVM", "KNN"],
    "Accuracy": [svm_accuracy, knn_accuracy]
})

print("\nModel Comparison")
print(results)

plt.figure(figsize=(5,4))
plt.bar(results["Model"], results["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.close()

# ============================================
# MLflow
# ============================================

mlflow.set_experiment("Week3_SVM_KNN")

with mlflow.start_run():

    mlflow.log_param("SVM Kernel", "rbf")
    mlflow.log_param("KNN Neighbors", 5)

    mlflow.log_metric("SVM Accuracy", svm_accuracy)
    mlflow.log_metric("KNN Accuracy", knn_accuracy)

    mlflow.sklearn.log_model(svm, "SVM_Model")
    mlflow.sklearn.log_model(knn, "KNN_Model")

# ============================================
# Save Models
# ============================================

joblib.dump(svm, "svm_model.pkl")
joblib.dump(knn, "knn_model.pkl")

print("\nModels saved successfully!")
print("\nTask Completed Successfully!")