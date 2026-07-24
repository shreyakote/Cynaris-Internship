# ============================================
# Week 3 - Day 5
# Hyperparameter Tuning using
# GridSearchCV & RandomizedSearchCV
# ============================================

import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
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
# Random Forest Model
# ============================================

model = RandomForestClassifier(random_state=42)

# ============================================
# Hyperparameter Grid
# ============================================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10]
}

# ============================================
# Grid Search
# ============================================

print("\nRunning GridSearchCV...")

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters (GridSearch):")
print(grid_search.best_params_)

grid_model = grid_search.best_estimator_

grid_predictions = grid_model.predict(X_test)

grid_accuracy = accuracy_score(y_test, grid_predictions)

print("\nGridSearch Accuracy:", grid_accuracy)

print(classification_report(y_test, grid_predictions))

# ============================================
# Randomized Search
# ============================================

print("\nRunning RandomizedSearchCV...")

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=5,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest Parameters (RandomSearch):")
print(random_search.best_params_)

random_model = random_search.best_estimator_

random_predictions = random_model.predict(X_test)

random_accuracy = accuracy_score(y_test, random_predictions)

print("\nRandomSearch Accuracy:", random_accuracy)

print(classification_report(y_test, random_predictions))

# ============================================
# Comparison
# ============================================

results = pd.DataFrame({
    "Method": ["GridSearchCV", "RandomizedSearchCV"],
    "Accuracy": [grid_accuracy, random_accuracy]
})

print("\nComparison")
print(results)

plt.figure(figsize=(6,4))
plt.bar(results["Method"], results["Accuracy"])
plt.title("Hyperparameter Tuning Comparison")
plt.ylabel("Accuracy")
plt.savefig("hyperparameter_comparison.png")
plt.close()

# ============================================
# Confusion Matrix
# ============================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    grid_predictions
)

plt.title("GridSearch Confusion Matrix")
plt.savefig("grid_confusion_matrix.png")
plt.close()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    random_predictions
)

plt.title("RandomSearch Confusion Matrix")
plt.savefig("random_confusion_matrix.png")
plt.close()

# ============================================
# MLflow Logging
# ============================================

mlflow.set_experiment("Week3_Hyperparameter_Tuning")

with mlflow.start_run():

    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric("Grid Accuracy", grid_accuracy)

    mlflow.log_metric("Random Accuracy", random_accuracy)

    mlflow.sklearn.log_model(
        grid_model,
        "GridSearch_Model"
    )

    mlflow.sklearn.log_model(
        random_model,
        "RandomSearch_Model"
    )

# ============================================
# Save Models
# ============================================

joblib.dump(
    grid_model,
    "best_grid_model.pkl"
)

joblib.dump(
    random_model,
    "best_random_model.pkl"
)

print("\nModels Saved Successfully!")
print("\nTask Completed Successfully!")