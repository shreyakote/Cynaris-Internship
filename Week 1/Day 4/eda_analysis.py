import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("indian_students.csv")

# -----------------------------
# Basic Information
# -----------------------------
print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Observation 1
# -----------------------------
print("\nObservation 1:")
print("Dataset has", df.shape[0], "rows and", df.shape[1], "columns.")

# Observation 2
print("\nObservation 2:")
print("No missing values are present." if df.isnull().sum().sum()==0 else "Missing values exist.")

# Observation 3
print("\nObservation 3:")
print("Courses:", df["Course"].unique())

# Observation 4
print("\nObservation 4:")
print("States:", df["State"].unique())

# Observation 5
print("\nObservation 5:")
print("Average Marks:", df["Marks"].mean())

# -----------------------------
# Distribution Plot
# -----------------------------
plt.figure(figsize=(6,4))
df["Marks"].hist()
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.savefig("marks_distribution.png")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
corr = df.corr(numeric_only=True)

plt.figure(figsize=(5,4))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")
plt.show()

# -----------------------------
# Top 10 Category Counts
# -----------------------------
plt.figure(figsize=(6,4))
df["Course"].value_counts().head(10).plot(kind="bar")
plt.title("Top Courses")
plt.xlabel("Course")
plt.ylabel("Count")
plt.savefig("top_courses.png")
plt.show()

print("\nEDA Completed Successfully!")