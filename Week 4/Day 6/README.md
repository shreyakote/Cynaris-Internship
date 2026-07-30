# Week 5 – Day 6

# Responsible AI & Fairness Analysis

## 📌 Project Overview

This project demonstrates Responsible AI principles by evaluating fairness in a machine learning model using the IBM AI Fairness 360 (AIF360) toolkit. The Adult Income dataset is used to identify gender bias, apply bias mitigation using Reweighing, and explain model predictions using SHAP.

---

## 🎯 Objectives

- Identify bias in machine learning datasets.
- Measure fairness using Disparate Impact.
- Train a Logistic Regression classifier.
- Apply Reweighing to reduce bias.
- Generate SHAP explanations for model predictions.
- Create documentation through a Model Card.
- Analyze a real-world AI bias case in India.

---

## 🛠 Technologies Used

- Python 3.11
- Scikit-learn
- IBM AI Fairness 360 (AIF360)
- SHAP
- Pandas
- NumPy
- Matplotlib

---

## 🤖 AI/ML 3M Stack

- **CrewAI** – Workflow planning
- **LangGraph** – Pipeline orchestration
- **MLflow** – Experiment tracking (optional)
- **Ragas** – Not applicable
- **MLOps** – Version control and reproducibility

---

## 📂 Project Structure

```
Week 5/
└── Day 6/
    ├── fairness_analysis.py
    ├── README.md
    ├── requirements.txt
    ├── model_card.md
    ├── india_ai_bias_case.md
    ├── shap_summary.png
    └── shap_waterfall.png
```

---

## 📊 Tasks Performed

- Loaded the Adult Income dataset.
- Computed Disparate Impact using AIF360.
- Applied Reweighing bias mitigation.
- Trained a Logistic Regression classifier.
- Evaluated the model using a Classification Report.
- Generated SHAP Summary Plot.
- Generated SHAP Waterfall Plot.
- Documented findings using a Model Card.

---

## 📈 Fairness Metric

**Disparate Impact**

- Ideal Value ≈ **1.0**
- Value **< 0.8** indicates possible bias.
- Reweighing is applied to improve fairness.

---

## 📷 Output Files

- shap_summary.png
- shap_waterfall.png

---

## ▶️ How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Execute

```bash
python fairness_analysis.py
```

---

## 📚 Learning Outcomes

- Understood six types of bias in AI.
- Measured fairness using IBM AIF360.
- Reduced bias using Reweighing.
- Explained predictions using SHAP.
- Learned Responsible AI documentation through Model Cards.

---

## 👩‍💻 Author

**K. Shreya**  
AI/ML Intern – Cynaries Solutions Pvt. Ltd.