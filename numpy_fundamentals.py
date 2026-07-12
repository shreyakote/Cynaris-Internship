import numpy as np
import pandas as pd

# 1D Array
arr1 = np.array([10, 20, 30, 40, 50])
print("1D Array:", arr1)
print("Shape:", arr1.shape)

# 2D Array
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
print("\n2D Array:")
print(arr2)
print("Shape:", arr2.shape)

# 3D Array
arr3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("\n3D Array:")
print(arr3)
print("Shape:", arr3.shape)

# Broadcasting
A = np.array([[1], [2], [3]])
B = np.array([10, 20, 30])

print("\nBroadcasting:")
print(A + B)

# Vectorised Operations
numbers = np.array([1, 2, 3, 4, 5])

print("\nSquare:", numbers ** 2)
print("Multiply by 5:", numbers * 5)

# Matrix Multiplication
M1 = np.array([[1, 2],
               [3, 4]])
M2 = np.array([[5, 6],
               [7, 8]])

print("\nMatrix Multiplication:")
print(M1 @ M2)
# -------------------------------
# CSV Statistics
# -------------------------------

import pandas as pd

df = pd.read_csv("student.csv")

print("\nStudent Dataset")
print(df)

print("\nMean")
print(df.mean())

print("\nStandard Deviation")
print(df.std())

print("\nCorrelation")
print(df.corr())