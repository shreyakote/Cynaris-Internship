import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import SelectKBest, f_classif

# Sample Dataset
data = {
    "Gender": ["Male", "Female", "Female", "Male", "Male"],
    "City": ["Bangalore", "Delhi", "Mumbai", "Delhi", "Mumbai"],
    "Education": ["High", "Medium", "Low", "Medium", "High"],
    "Age": [23, 35, 29, 40, 31],
    "Salary": [35000, 70000, 45000, 90000, 60000],
    "Purchased": [0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)
# ---------------- Label Encoding ----------------
le = LabelEncoder()

df["Gender_Label"] = le.fit_transform(df["Gender"])

print("\nAfter Label Encoding:")
print(df[["Gender", "Gender_Label"]])
# ---------------- One Hot Encoding ----------------
city_encoded = pd.get_dummies(df["City"], prefix="City")

print("\nOne Hot Encoded City:")
print(city_encoded)
# ---------------- Ordinal Encoding ----------------
oe = OrdinalEncoder(categories=[["Low", "Medium", "High"]])

df["Education_Ordinal"] = oe.fit_transform(df[["Education"]])

print("\nOrdinal Encoded Education:")
print(df[["Education", "Education_Ordinal"]])

# ---------------- Scaling ----------------

# Select numerical columns
numerical_data = df[["Age", "Salary"]]

# StandardScaler
standard_scaler = StandardScaler()
standard_scaled = pd.DataFrame(
    standard_scaler.fit_transform(numerical_data),
    columns=["Age", "Salary"]
)

print("\nStandardScaler Output:")
print(standard_scaled)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaled = pd.DataFrame(
    minmax_scaler.fit_transform(numerical_data),
    columns=["Age", "Salary"]
)

print("\nMinMaxScaler Output:")
print(minmax_scaled)

# RobustScaler
robust_scaler = RobustScaler()
robust_scaled = pd.DataFrame(
    robust_scaler.fit_transform(numerical_data),
    columns=["Age", "Salary"]
)

print("\nRobustScaler Output:")
print(robust_scaled)

# ---------------- Distribution Plots ----------------

plt.figure(figsize=(6,4))
plt.hist(df["Salary"], bins=5)
plt.title("Before Scaling")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(standard_scaled["Salary"], bins=5)
plt.title("After StandardScaler")
plt.xlabel("Scaled Salary")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(minmax_scaled["Salary"], bins=5)
plt.title("After MinMaxScaler")
plt.xlabel("Scaled Salary")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(robust_scaled["Salary"], bins=5)
plt.title("After RobustScaler")
plt.xlabel("Scaled Salary")
plt.ylabel("Frequency")
plt.show()

# ---------------- Feature Selection ----------------

# Combine encoded features
X = pd.concat([
    df[["Age", "Salary", "Gender_Label", "Education_Ordinal"]],
    city_encoded.astype(int)
], axis=1)

# Target column
y = df["Purchased"]

# Select top 5 features
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X, y)

# Create a DataFrame of feature scores
scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

# Sort features by score
scores = scores.sort_values(by="Score", ascending=False)

print("\nTop Features:")
print(scores)

print("\nTop 5 Features:")
print(scores.head(5))