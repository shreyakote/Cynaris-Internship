# MLOps Monitoring Strategy — Day 2

## 1. Overview

The Day 2 ML API is a Dockerised FastAPI application that serves a Random Forest classification model.

Monitoring is required to ensure that the API remains available, performs efficiently, and continues to produce reliable predictions.

## 2. Metrics to Monitor

| Metric | Purpose |
|---|---|
| Request count | Track API usage |
| Response latency | Measure prediction response time |
| Error rate | Detect failed requests |
| HTTP status codes | Identify successful and failed requests |
| CPU usage | Monitor container CPU consumption |
| Memory usage | Detect excessive memory consumption |
| Prediction distribution | Detect unusual changes in model outputs |
| Input data distribution | Detect changes in incoming data |
| Model accuracy | Measure model quality when labelled data is available |
| Data drift | Detect changes between training and production data |

## 3. Alerts

The following conditions can generate alerts:

- API error rate exceeds the configured threshold.
- Prediction response latency becomes too high.
- The health endpoint repeatedly fails.
- Container CPU or memory usage becomes excessive.
- Significant data drift is detected.
- Model accuracy falls below the agreed threshold.

Alert thresholds should be adjusted using actual production measurements.

## 4. Retraining Triggers

Model retraining should be considered when:

1. Model performance decreases below the required threshold.
2. Significant data drift is detected.
3. New labelled training data becomes available.
4. Prediction quality consistently decreases.
5. A scheduled model review identifies the need for an updated model.

A model should be evaluated and tested before replacing the currently deployed model.

## 5. Monitoring Tools

The MLOps workflow can use:

- FastAPI health endpoint for API availability.
- Docker for container status and resource monitoring.
- Python logging for application events and errors.
- MLflow for experiment tracking and model versions.
- Prometheus and Grafana for metrics and dashboards.
- Ragas for evaluating RAG systems when applicable.

## 6. Incident Response

When an alert occurs:

1. Check the API and container status.
2. Inspect application logs.
3. Identify whether the issue is related to infrastructure, API errors, input data, or model performance.
4. Apply the required fix.
5. Run tests after the fix.
6. Document the incident and preventive action.

## 7. Conclusion

An effective MLOps monitoring strategy combines API monitoring, infrastructure monitoring, model performance tracking, data drift detection, alerts, and controlled retraining.

This helps maintain a reliable ML service after deployment.