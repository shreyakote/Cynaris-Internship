import pandas as pd

# Load the dataset
df = pd.read_csv("indian_students.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display shape
print("\nShape:")
print(df.shape)

# Display column names
print("\nColumns:")
print(df.columns)

# Display data types
print("\nData Types:")
print(df.dtypes)

# Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values in Marks column with the average
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

# Remove duplicate rows
df = df.drop_duplicates()

# Rename StudentID to ID
df.rename(columns={"StudentID": "ID"}, inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nData cleaned successfully!")