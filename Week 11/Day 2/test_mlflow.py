import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_uri = "models:/Day2IrisRandomForest/1"

model = mlflow.sklearn.load_model(model_uri)

sample_data = [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5],
]

predictions = model.predict(sample_data)

print("=" * 50)
print("REGISTERED MODEL LOADED SUCCESSFULLY")
print("=" * 50)
print(f"Model URI: {model_uri}")
print(f"Predictions: {predictions}")
print("=" * 50)
