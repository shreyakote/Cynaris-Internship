# Monitoring ML Models in Production

## 1. Overview

The ML application is deployed as a Dockerised FastAPI service using a Random Forest classification model.

Production monitoring helps track API reliability, infrastructure usage, input data changes, and model performance.

## 2. Metrics to Track

| Metric | What to Monitor |
|---|---|
| Request count | Number of prediction requests |
| Response latency | Time taken to return predictions |
| Error rate | Percentage of failed requests |
| HTTP status codes | Successful and failed API requests |
| CPU usage | Container CPU consumption |
| Memory usage | Container memory consumption |
| Prediction distribution | Changes in predicted classes |
| Input data distribution | Changes in incoming feature values |
| Data drift | Difference between training and production data |
| Model accuracy | Prediction quality when labelled data is available |

## 3. Alerts

The following alerts should be configured:

- Alert when API error rate exceeds the configured threshold.
- Alert when response latency becomes consistently high.
- Alert when the health endpoint fails repeatedly.
- Alert when CPU or memory usage becomes excessive.
- Alert when significant data drift is detected.
- Alert when model accuracy falls below the required threshold.

Initial thresholds should be adjusted using real production traffic and baseline measurements.

## 4. Retraining Triggers

Model retraining should be considered when:

1. Model accuracy falls below the agreed threshold.
2. Significant data drift is detected.
3. New labelled data becomes available.
4. Prediction quality decreases consistently.
5. A scheduled model review identifies the need for a new model.

A single alert should not automatically replace the production model. The new model should first be trained, evaluated, tested, and approved.

## 5. Monitoring Tools

The monitoring workflow can use:

- FastAPI health endpoint for service availability.
- Docker for container status, logs, and resource usage.
- Python logging for application events and errors.
- MLflow for experiment tracking and model versions.
- Prometheus and Grafana for metrics, dashboards, and alerts.
- Ragas for evaluating RAG systems where applicable.

## 6. Incident Response

When an alert occurs:

1. Check API and container status.
2. Inspect application and Docker logs.
3. Determine whether the problem is infrastructure, API errors, input data, or model performance.
4. Apply the appropriate fix.
5. Run automated tests.
6. Re-evaluate the model if model quality is affected.
7. Document the incident and preventive action.

## 7. Conclusion

Production ML monitoring combines application monitoring, infrastructure monitoring, data drift detection, model performance tracking, alerts, and controlled retraining.

This helps maintain a reliable ML service and provides a process for detecting and responding to model degradation.