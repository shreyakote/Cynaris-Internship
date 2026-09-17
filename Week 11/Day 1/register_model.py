import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Create/select experiment
mlflow.set_experiment("Iris Random Forest Experiments")


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


# Use the best configuration from our five experiments
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=3,
    random_state=42,
)


# Start MLflow run
with mlflow.start_run() as run:

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
        50,
    )

    mlflow.log_param(
        "max_depth",
        3,
    )

    # Log metric
    mlflow.log_metric(
        "accuracy",
        accuracy,
    )

    # Log model
    model_info = mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        signature=signature,
    )

    print("=" * 50)
    print("MODEL CREATED SUCCESSFULLY")
    print("=" * 50)
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model URI: {model_info.model_uri}")
    print("=" * 50)