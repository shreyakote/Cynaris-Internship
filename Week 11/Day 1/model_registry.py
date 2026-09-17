import mlflow
import mlflow.sklearn


# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Model URI created by MLflow
model_uri = "models:/m-48ee643f56444485a3a121c8f1d00cc7"


# Register the model
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="IrisRandomForest",
)


print("=" * 50)
print("MODEL REGISTERED SUCCESSFULLY")
print("=" * 50)
print(f"Model name: {registered_model.name}")
print(f"Model version: {registered_model.version}")
print("=" * 50)


# Load the registered model
registered_model_uri = (
    f"models:/IrisRandomForest/{registered_model.version}"
)

loaded_model = mlflow.sklearn.load_model(
    registered_model_uri
)


# Test the loaded model
sample_data = [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5],
]

predictions = loaded_model.predict(sample_data)


print("REGISTERED MODEL LOADED SUCCESSFULLY")
print(f"Predictions: {predictions}")
print("=" * 50)