
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Configure MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Week 11 MLflow Experiments")

# Five hyperparameter configurations
experiments = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]

for index, params in enumerate(experiments, start=1):
    with mlflow.start_run(run_name=f"Experiment_{index}"):

        model = RandomForestClassifier(
            **params,
            random_state=42,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log hyperparameters
        mlflow.log_params(params)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
        )

        print(f"\nExperiment {index}")
        print(f"Parameters: {params}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

print("\nAll five experiments completed!")