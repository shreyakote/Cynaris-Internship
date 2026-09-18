import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

run_id = "b685af61f69c49adbad200a7bde4cb2e"

model_uri = f"runs:/{run_id}/random_forest_model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="Day2IrisRandomForest",
)

print("=" * 50)
print("MODEL REGISTERED SUCCESSFULLY")
print("=" * 50)
print(f"Model name: {registered_model.name}")
print(f"Model version: {registered_model.version}")
print("=" * 50)
