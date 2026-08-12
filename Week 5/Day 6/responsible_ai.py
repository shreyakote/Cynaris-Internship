import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import shap
import matplotlib.pyplot as plt

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


print("=" * 60)
print("RESPONSIBLE AI & FAIRNESS ANALYSIS")
print("=" * 60)


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATASET
# ---------------------------------------------------------

print("\nLoading dataset...")

np.random.seed(42)

n_samples = 1000

data = pd.DataFrame({
    "age": np.random.randint(20, 60, n_samples),
    "education_years": np.random.randint(8, 18, n_samples),
    "hours_per_week": np.random.randint(20, 60, n_samples),
    "experience": np.random.randint(0, 30, n_samples),
    "sex": np.random.randint(0, 2, n_samples)
})

# Create income label
data["income"] = (
    (data["education_years"] >= 13) &
    (data["hours_per_week"] >= 35) &
    (data["experience"] >= 5)
).astype(int)

print("Dataset created successfully!")
print("Number of records:", len(data))


# ---------------------------------------------------------
# 2. FEATURES AND TARGET
# ---------------------------------------------------------

features = [
    "age",
    "education_years",
    "hours_per_week",
    "experience"
]

X = data[features]
y = data["income"]

protected_attribute = data["sex"]


# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X,
    y,
    protected_attribute,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 4. SCALE FEATURES
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", round(accuracy, 4))


# ---------------------------------------------------------
# 6. CREATE AIF360 DATASETS
# ---------------------------------------------------------

test_data = X_test.copy()

test_data["sex"] = sex_test.values
test_data["income"] = y_test.values

predicted_data = X_test.copy()

predicted_data["sex"] = sex_test.values
predicted_data["income"] = predictions


dataset_true = BinaryLabelDataset(
    df=test_data,
    label_names=["income"],
    protected_attribute_names=["sex"],
    favorable_label=1,
    unfavorable_label=0
)

dataset_pred = BinaryLabelDataset(
    df=predicted_data,
    label_names=["income"],
    protected_attribute_names=["sex"],
    favorable_label=1,
    unfavorable_label=0
)


# ---------------------------------------------------------
# 7. FAIRNESS METRICS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FAIRNESS METRICS")
print("=" * 60)

metric = ClassificationMetric(
    dataset_true,
    dataset_pred,
    unprivileged_groups=[{"sex": 0}],
    privileged_groups=[{"sex": 1}]
)


spd = metric.statistical_parity_difference()

di = metric.disparate_impact()

eod = metric.equal_opportunity_difference()


print("\n1. Statistical Parity Difference:")
print(round(spd, 4))

print("\n2. Disparate Impact:")
print(round(di, 4))

print("\n3. Equal Opportunity Difference:")
print(round(eod, 4))


# ---------------------------------------------------------
# 8. SAVE FAIRNESS RESULTS
# ---------------------------------------------------------

with open("fairness_metrics.txt", "w") as file:

    file.write("RESPONSIBLE AI FAIRNESS ANALYSIS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Model Accuracy: {accuracy:.4f}\n\n")

    file.write(
        f"Statistical Parity Difference: {spd:.4f}\n"
    )

    file.write(
        f"Disparate Impact: {di:.4f}\n"
    )

    file.write(
        f"Equal Opportunity Difference: {eod:.4f}\n"
    )

print("\nFairness results saved to fairness_metrics.txt")


# ---------------------------------------------------------
# 9. SHAP EXPLANATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SHAP EXPLANATION")
print("=" * 60)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=features
)

explainer = shap.LinearExplainer(
    model,
    X_train_scaled
)

shap_values = explainer.shap_values(
    X_test_scaled
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_scaled_df,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP explanation saved to shap_summary.png")


# ---------------------------------------------------------
# 10. CREATE MODEL CARD
# ---------------------------------------------------------

model_card = f"""
# Model Card - Income Classification Model

## Model Overview

Model type: Logistic Regression

Purpose:
Predict whether an individual belongs to the higher-income
class based on selected demographic and employment-related
features.

## Features

- Age
- Education years
- Hours per week
- Experience

## Protected Attribute

Sex

## Dataset

Synthetic demonstration dataset created for responsible AI
and fairness analysis.

## Performance

Accuracy: {accuracy:.4f}

## Fairness Metrics

Statistical Parity Difference: {spd:.4f}

Disparate Impact: {di:.4f}

Equal Opportunity Difference: {eod:.4f}

## Explainability

SHAP was used to understand the contribution of each feature
to model predictions.

## Limitations

This model is intended only for educational demonstration.
It should not be used for real-world employment, lending,
insurance, healthcare, or other high-impact decisions.

Fairness metrics should be evaluated using representative
real-world data before deployment.
"""

with open("model_card.md", "w") as file:
    file.write(model_card)

print("Model card saved to model_card.md")


print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)