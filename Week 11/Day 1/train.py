import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load Iris dataset
X, y = load_iris(return_X_y=True)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Create MLflow experiment
mlflow.set_experiment("Iris Random Forest Experiments")


# Five experiments
experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 5},
    {"n_estimators": 200, "max_depth": None},
]


# Run experiments
for experiment in experiments:

    n_estimators = experiment["n_estimators"]
    max_depth = experiment["max_depth"]

    with mlflow.start_run() as run:

        # Create model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        # Train model
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        # Create model signature
        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        # Log parameters
        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )

        mlflow.log_param(
            "max_depth",
            max_depth,
        )

        # Log metric
        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        # Log model
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            signature=signature,
        )

        # Display result
        print("=" * 50)
        print(f"Run ID: {run.info.run_id}")
        print(f"n_estimators: {n_estimators}")
        print(f"max_depth: {max_depth}")
        print(f"accuracy: {accuracy:.4f}")
        print("=" * 50)