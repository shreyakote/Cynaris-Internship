
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

Accuracy: 0.8800

## Fairness Metrics

Statistical Parity Difference: 0.0303

Disparate Impact: 1.1379

Equal Opportunity Difference: 0.0472

## Explainability

SHAP was used to understand the contribution of each feature
to model predictions.

## Limitations

This model is intended only for educational demonstration.
It should not be used for real-world employment, lending,
insurance, healthcare, or other high-impact decisions.

Fairness metrics should be evaluated using representative
real-world data before deployment.
