import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Create Sample Dataset
# -----------------------------
data = {
    "Age": [25, 30, np.nan, 28, 40, 35, np.nan, 45],
    "Salary": [50000, 60000, 55000, np.nan, 70000, 65000, 62000, 72000],
    "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male"],
    "Purchased": [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Features and Target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Numerical and Categorical Columns
numeric_features = ["Age", "Salary"]
categorical_features = ["Gender"]

# Numeric Pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])

# Combine Pipelines
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Transform Data
X_processed = preprocessor.fit_transform(X)

print("\nProcessed Data:")
print(X_processed)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

print("\nPipeline completed successfully!")