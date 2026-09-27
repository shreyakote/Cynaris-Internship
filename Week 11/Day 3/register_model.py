
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("file:./mlruns")

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "Week 11 MLflow Experiments"
)

if experiment is None:
    raise ValueError("Experiment not found.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
)

if not runs:
    raise ValueError("No experiment runs found.")

# Select the first run with the highest accuracy
best_run = runs[0]

model_uri = f"runs:/{best_run.info.run_id}/model"

model_name = "Week11RandomForestModel"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)

print("Model registered successfully!")
print(f"Model name: {model_name}")
print(f"Model version: {registered_model.version}")
print(f"Run ID: {best_run.info.run_id}")
print(f"Accuracy: {best_run.data.metrics['accuracy']}")