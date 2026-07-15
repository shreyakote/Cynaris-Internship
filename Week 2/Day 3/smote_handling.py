import pandas as pd
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ------------------------------------
# Create an Imbalanced Dataset
# ------------------------------------

X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    weights=[0.9, 0.1],
    random_state=42
)

print("Class distribution before SMOTE:")
print(Counter(y))

# ------------------------------------
# Split Dataset
# ------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------
# Apply SMOTE
# ------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(Counter(y_train_smote))

# ------------------------------------
# Save balanced dataset
# ------------------------------------

balanced_df = pd.DataFrame(X_train_smote)
balanced_df["Target"] = y_train_smote

balanced_df.to_csv("balanced_dataset.csv", index=False)

print("\nBalanced dataset saved successfully.")