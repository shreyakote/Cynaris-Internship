
import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer

# Load the same dataset used for training
data = load_breast_cancer()

# Select one sample with all 30 features
sample = pd.DataFrame(
    data.data[:1],
    columns=data.feature_names,
)

# Prepare the MLflow REST API request
payload = {
    "dataframe_split": sample.to_dict(orient="split")
}

# Send the prediction request
response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json=payload,
    timeout=60,
)

print("Status code:", response.status_code)
print("Prediction response:", response.text)

response.raise_for_status()

print("Prediction API test completed successfully!")