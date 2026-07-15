import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.feature_selection import SelectKBest, f_classif

# ----------------------------
# Sample Dataset
# ----------------------------

df = pd.DataFrame({
    "Gender": ["Male","Female","Female","Male","Female","Male","Female","Male"],
    "Department": ["IT","HR","Finance","IT","HR","Finance","IT","HR"],
    "Performance": ["Low","Medium","High","Medium","High","Low","Medium","High"],
    "Age": [21,22,20,24,23,25,22,26],
    "Salary": [30000,35000,40000,45000,42000,50000,38000,52000],
    "Experience": [1,2,1,3,2,4,2,5],
    "Selected": [0,1,1,1,1,0,1,0]
})

print("\nOriginal Dataset\n")
print(df)

# ----------------------------
# Label Encoding
# ----------------------------

label = LabelEncoder()

df["Gender_Label"] = label.fit_transform(df["Gender"])

print("\nLabel Encoding")
print(df[["Gender","Gender_Label"]])

# ----------------------------
# One Hot Encoding
# ----------------------------

onehot = pd.get_dummies(df["Department"], prefix="Dept")

print("\nOne Hot Encoding")
print(onehot)

# ----------------------------
# Ordinal Encoding
# ----------------------------

ordinal = OrdinalEncoder(
    categories=[["Low","Medium","High"]]
)

df["Performance_Ordinal"] = ordinal.fit_transform(
    df[["Performance"]]
)

print("\nOrdinal Encoding")
print(df[["Performance","Performance_Ordinal"]])

# ----------------------------
# Feature Scaling
# ----------------------------

numeric = df[["Age","Salary","Experience"]]

# StandardScaler
standard = StandardScaler()
standard_scaled = pd.DataFrame(
    standard.fit_transform(numeric),
    columns=numeric.columns
)

# MinMaxScaler
minmax = MinMaxScaler()
minmax_scaled = pd.DataFrame(
    minmax.fit_transform(numeric),
    columns=numeric.columns
)

# RobustScaler
robust = RobustScaler()
robust_scaled = pd.DataFrame(
    robust.fit_transform(numeric),
    columns=numeric.columns
)

print("\nStandard Scaled")
print(standard_scaled)

print("\nMinMax Scaled")
print(minmax_scaled)

print("\nRobust Scaled")
print(robust_scaled)

# ----------------------------
# Plot Before Scaling
# ----------------------------

plt.figure(figsize=(6,4))
plt.hist(df["Salary"], bins=5)
plt.title("Salary Before Scaling")
plt.xlabel("Salary")
plt.ylabel("Count")
plt.savefig("before_scaling.png")
plt.close()

# ----------------------------
# Plot After Scaling
# ----------------------------

plt.figure(figsize=(6,4))
plt.hist(standard_scaled["Salary"], bins=5)
plt.title("Salary After Standard Scaling")
plt.xlabel("Scaled Salary")
plt.ylabel("Count")
plt.savefig("after_scaling.png")
plt.close()

# ----------------------------
# Feature Selection
# ----------------------------

features = pd.concat([
    standard_scaled,
    df[["Gender_Label","Performance_Ordinal"]],
    onehot
], axis=1)

target = df["Selected"]

selector = SelectKBest(score_func=f_classif, k=5)

X_new = selector.fit_transform(features, target)

selected = features.columns[selector.get_support()]

print("\nTop 5 Features")
print(selected)

scores = pd.DataFrame({
    "Feature": features.columns,
    "Score": selector.scores_
})

print("\nFeature Scores")
print(scores.sort_values("Score", ascending=False))