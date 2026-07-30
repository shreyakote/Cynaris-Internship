# Model Card

## Model Name
Adult Income Classification using Logistic Regression

## Model Type
Binary Classification

## Developer
Shreya K

## Intended Use
Predict whether an individual's annual income exceeds $50,000.

## Dataset
Adult Income Dataset (UCI Machine Learning Repository)

## Features
- Age
- Education
- Occupation
- Hours per Week
- Marital Status
- Race
- Gender
- Capital Gain
- Capital Loss
- Native Country

## Evaluation Metrics

Accuracy: 84.47%

Disparate Impact Before Reweighing:
0.3635

Disparate Impact After Reweighing:
1.0000

## Explainability
SHAP was used to explain feature importance and individual predictions.

## Ethical Considerations
The original dataset showed gender bias. Reweighing was applied to reduce bias and improve fairness.

## Limitations
- Dataset is based on U.S. Census data.
- Results may not generalize to other populations.
- Only gender fairness was evaluated.

## Conclusion
The model achieves good predictive performance while significantly improving fairness using AIF360's Reweighing algorithm.