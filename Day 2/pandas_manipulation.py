import pandas as pd
import os

# -----------------------------
# Load CSV Dataset
# -----------------------------
df = pd.read_csv("indian_students.csv")

# Print shape
print("Shape:")
print(df.shape)

# Print data types
print("\nData Types:")
print(df.dtypes)

# Print first 10 rows
print("\nFirst 10 Rows:")
print(df.head(10))

# -----------------------------
# Filter Operation
# Students with Marks > 85
# -----------------------------
filtered_df = df[df["Marks"] > 85]

print("\nFiltered Data")
print(filtered_df)

# -----------------------------
# GroupBy Operation
# Average marks by State
# -----------------------------
grouped_df = df.groupby("State")["Marks"].mean()

print("\nAverage Marks by State")
print(grouped_df)

# -----------------------------
# Merge Operation
# Add course duration
# -----------------------------
course_df = pd.DataFrame({
    "Course": ["AIML", "Data Science", "Cyber Security"],
    "Duration": ["4 Months", "6 Months", "5 Months"]
})

merged_df = pd.merge(df, course_df, on="Course")

print("\nMerged Data")
print(merged_df)

# -----------------------------
# Pivot Table
# Average marks by State and Course
# -----------------------------
pivot = df.pivot_table(
    values="Marks",
    index="State",
    columns="Course",
    aggfunc="mean"
)

print("\nPivot Table")
print(pivot)

# -----------------------------
# Export cleaned data
# -----------------------------
merged_df.to_csv("cleaned_students.csv", index=False)
merged_df.to_parquet("cleaned_students.parquet", index=False)

# -----------------------------
# Compare file sizes
# -----------------------------
csv_size = os.path.getsize("cleaned_students.csv")
parquet_size = os.path.getsize("cleaned_students.parquet")

print("\nCSV File Size:", csv_size, "bytes")
print("Parquet File Size:", parquet_size, "bytes")