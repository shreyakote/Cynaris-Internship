import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Create Sample Dataset
# -----------------------------
np.random.seed(42)

data = pd.DataFrame({
    "Age": np.random.randint(18, 60, 100),
    "Salary": np.random.randint(25000, 100000, 100),
    "Experience": np.random.randint(0, 20, 100),
    "Selected": np.random.randint(0, 2, 100)
})

print("First 5 Rows")
print(data.head())

# -----------------------------
# Features and Target
# -----------------------------
X = data[["Age", "Salary", "Experience"]]
y = data["Selected"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 2))

# -----------------------------
# Cross Validation
# -----------------------------
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")
print(scores)

print("\nAverage Accuracy")
print(scores.mean())