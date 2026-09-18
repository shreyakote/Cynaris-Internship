import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Day 2 - Iris Random Forest")

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

experiments = [
    (50, 3),
    (100, 3),
    (100, 5),
    (150, 5),
    (200, None),
]

for n_estimators, max_depth in experiments:
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            signature=signature,
            pip_requirements=[
                "mlflow",
                "scikit-learn",
            ],
        )

        print(
            f"Run ID: {run.info.run_id} | "
            f"n_estimators={n_estimators} | "
            f"max_depth={max_depth} | "
            f"accuracy={accuracy:.4f}"
        )
