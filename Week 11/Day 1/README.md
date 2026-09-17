# MLflow Experiment Tracking and Model Registry

## Objective

Track machine learning experiments using MLflow, log model parameters and metrics, register the trained model, and serve the registered model through an API.

## Model

- Dataset: Iris
- Algorithm: RandomForestClassifier
- Framework: Scikit-learn
- Experiment: Iris Random Forest Experiments

## Logged Parameters

- model_type
- n_estimators
- max_depth

## Logged Metric

- accuracy

## Experiments

Five Random Forest configurations were tested with different hyperparameters.

The best configuration used:

- n_estimators: 50
- max_depth: 3
- accuracy: 0.9667

## MLflow Model Registry

The best model was registered as:

IrisRandomForest

Model version:

1

## Model Serving

The registered model was served using:

mlflow models serve

The model API ran on:

http://127.0.0.1:5001

## API Test

The served model was tested using Iris feature inputs and returned predictions successfully.

## Technologies

- MLflow
- Scikit-learn
- Pandas
- Pytest
- Ruff
