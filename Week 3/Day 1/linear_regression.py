import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load California Housing Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([name, mse, rmse, mae, r2])

    print(f"\n{name}")
    print("Coefficients:")
    print(model.coef_)

    if name == "Linear Regression":

        plt.figure(figsize=(6,5))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        plt.savefig("predicted_vs_actual.png")
        plt.close()

        residuals = y_test - y_pred

        plt.figure(figsize=(6,5))
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color="red")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.savefig("residual_plot.png")
        plt.close()

results_df = pd.DataFrame(
    results,
    columns=["Model", "MSE", "RMSE", "MAE", "R²"]
)

print("\nModel Comparison")
print(results_df)

results_df.to_csv("results.csv", index=False)

print("\nResults saved successfully!")