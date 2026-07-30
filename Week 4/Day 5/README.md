# Week 4 - Day 5

# 1M Capstone: Sentiment Classifier – Deploy & Document

## 📌 Project Overview

This project implements a **Sentiment Analysis Classifier** using the NLTK Movie Reviews dataset. It compares the performance of **Logistic Regression** and **Random Forest Classifier** for binary sentiment classification (Positive/Negative).

The project follows the approved **AI/ML 3M Stack** by incorporating MLflow for experiment tracking and MLOps best practices for reproducibility and version control.

---

## 🎯 Objectives

- Build a sentiment classification model.
- Convert text into numerical features using TF-IDF.
- Train Logistic Regression and Random Forest models.
- Compare model performance.
- Evaluate using multiple classification metrics.
- Visualize Confusion Matrix and ROC-AUC Curve.
- Track experiments using MLflow.

---

## 🛠 Technologies Used

- Python
- Scikit-learn
- NLTK
- MLflow
- Matplotlib
- NumPy

---

## 🤖 AI/ML 3M Stack

- **CrewAI** – Workflow documentation
- **LangGraph** – Pipeline planning
- **MLflow** – Experiment tracking
- **Ragas** – Not required for this task
- **MLOps** – Version control, reproducibility, and model management

---

## 📂 Project Structure

```
Week 4/
└── Day 5/
    ├── sentiment_classifier.py
    ├── README.md
    ├── requirements.txt
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    └── roc_curve_comparison.png
```

---

## 📊 Models Used

1. Logistic Regression
2. Random Forest Classifier

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- Classification Report
- Confusion Matrix
- ROC-AUC Score

---

## 📷 Output Files

- confusion_matrix_logistic_regression.png
- confusion_matrix_random_forest.png
- roc_curve_comparison.png

---

## ▶️ How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python sentiment_classifier.py
```

### View MLflow Dashboard

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## ✅ Results

The project successfully:

- Performed sentiment classification on movie reviews.
- Compared Logistic Regression and Random Forest models.
- Generated classification reports.
- Created confusion matrices for both models.
- Plotted ROC-AUC comparison curves.
- Logged evaluation metrics and artifacts using MLflow.

---

## 📚 Learning Outcomes

- Learned text preprocessing using TF-IDF.
- Built sentiment analysis models using Scikit-learn.
- Compared multiple classification algorithms.
- Evaluated models using standard classification metrics.
- Tracked experiments with MLflow.
- Applied basic MLOps practices for reproducibility.

---

## 👩‍💻 Author

**K. Shreya**  
AI/ML Intern – Cynaries Solutions Pvt. Ltd.