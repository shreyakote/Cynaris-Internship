"""
Week 4 Day 2
Bias-Variance Tradeoff & Regularisation

AI/ML 3M Stack:
- MLflow
- MLOps
"""

import numpy as np
import matplotlib.pyplot as plt
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------
# Generate Synthetic Dataset
# -------------------------------------------------

np.random.seed(42)

X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.15, 200)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# -------------------------------------------------
# MLflow
# -------------------------------------------------

mlflow.set_experiment("Week4_Day2_Bias_Variance")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01)
}

results = {}

with mlflow.start_run():

    plt.figure(figsize=(10,6))

    plt.scatter(X, y, color="gray", alpha=0.4, label="Data")

    x_plot = np.linspace(-3,3,300).reshape(-1,1)

    for name, model in models.items():

        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=8)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        prediction = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)

        results[name] = [mse, r2]

        mlflow.log_metric(f"{name}_MSE", mse)
        mlflow.log_metric(f"{name}_R2", r2)

        y_plot = pipeline.predict(x_plot)

        plt.plot(x_plot, y_plot, label=name)

    plt.title("Bias-Variance Tradeoff using Regularisation")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

    plt.savefig("polynomial_fit.png")
    mlflow.log_artifact("polynomial_fit.png")

    plt.close()

# -------------------------------------------------
# Print Results
# -------------------------------------------------

print("\nModel Comparison\n")

for model, values in results.items():
    print(f"{model}")
    print(f"MSE : {values[0]:.4f}")
    print(f"R2  : {values[1]:.4f}")
    print("-"*30)

# -------------------------------------------------
# Comparison Chart
# -------------------------------------------------

names = list(results.keys())
mse_values = [results[m][0] for m in names]

plt.figure(figsize=(7,5))
plt.bar(names, mse_values)

plt.title("Regularisation Comparison")
plt.ylabel("Mean Squared Error")

plt.savefig("regularization_comparison.png")

print("\nCompleted Successfully!")