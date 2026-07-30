"""
Week 5 Day 6 - Responsible AI & Fairness
Compatible with:
- Python 3.11
- aif360 0.6.1
- shap 0.51.0
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

print("="*60)
print("RESPONSIBLE AI & FAIRNESS ANALYSIS")
print("="*60)

# Load Adult dataset shipped with AIF360
dataset = AdultDataset()

metric = BinaryLabelDatasetMetric(
    dataset,
    privileged_groups=[{"sex": 1.0}],
    unprivileged_groups=[{"sex": 0.0}]
)

print("\nDisparate Impact (Before):", metric.disparate_impact())

rw = Reweighing(
    privileged_groups=[{"sex": 1.0}],
    unprivileged_groups=[{"sex": 0.0}]
)

dataset_rw = rw.fit_transform(dataset)

metric_after = BinaryLabelDatasetMetric(
    dataset_rw,
    privileged_groups=[{"sex": 1.0}],
    unprivileged_groups=[{"sex": 0.0}]
)

print("Disparate Impact (After):", metric_after.disparate_impact())

X = dataset.features
y = dataset.labels.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report")
print(classification_report(y_test, pred))

explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values.values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()

shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("shap_waterfall.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGenerated:")
print("- shap_summary.png")
print("- shap_waterfall.png")
print("\nCompleted successfully.")
