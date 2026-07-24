# Week 3 - Day 3
## Decision Trees & Random Forests

### Objective
The objective of this task is to understand and implement Decision Tree and Random Forest classification algorithms using Scikit-learn. The models are trained, evaluated, and compared using a real-world dataset. MLflow is used for experiment tracking, and the trained model is saved for future use.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- MLflow
- Joblib

---

## AI/ML 3M Stack

- **CrewAI:** Used for code review and guidance.
- **LangGraph:** Planned for workflow orchestration in AI applications.
- **MLflow:** Used for experiment tracking and model logging.
- **Ragas:** Mentioned as part of the approved AI/ML stack for evaluating RAG systems.
- **MLOps:** Git version control, reproducible code, and model management.

---

## Dataset

- Breast Cancer Wisconsin Dataset
- Source: `sklearn.datasets.load_breast_cancer()`

---

## Models Implemented

### 1. Decision Tree Classifier
- Trained using the training dataset.
- Predictions generated on the test dataset.
- Evaluated using classification metrics.

### 2. Random Forest Classifier
- Trained with 100 decision trees.
- Compared against the Decision Tree model.
- Feature importance analyzed.

---

## Evaluation Metrics

- Accuracy Score
- Classification Report
- Confusion Matrix

---

## Output Files

- `decision_tree_random_forest.py`
- `feature_importance.png`
- `confusion_matrix_dt.png`
- `confusion_matrix_rf.png`
- `random_forest_model.pkl`

---

## How to Run

1. Activate the virtual environment.

```bash
.\venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install pandas numpy matplotlib scikit-learn mlflow joblib
```

3. Run the program.

```bash
python decision_tree_random_forest.py
```

---

## Results

- Successfully trained Decision Tree and Random Forest models.
- Compared their classification performance.
- Generated confusion matrices.
- Visualized feature importance.
- Saved the trained Random Forest model.
- Logged the experiment using MLflow.

---

## Git Workflow

Branch:
```
feat/aiml-W3-shreya
```

Example commits:

```
feat: implement Decision Tree classifier and evaluation

feat: add Random Forest model with MLflow tracking and feature importance visualization
```

---

## Conclusion

This task demonstrated the implementation and evaluation of Decision Tree and Random Forest algorithms. Random Forest generally achieved better performance due to its ensemble learning approach. MLflow was used to track experiments, and the trained model was saved for future deployment.