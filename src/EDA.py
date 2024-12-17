import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data/Customer_Churn.csv')

# Basic information about the dataset
print("\n=== Dataset Info ===")
print(df.info())  # Shows data types and missing values

# Summary statistics for all numeric columns
print("\n=== Summary Statistics ===")
print(df.describe())  # Shows count, mean, std, min, 25%, 50%, 75%, max

# Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Display first few rows
print("\n=== First Few Rows ===")
print(df.head())

# Get column names
print("\n=== Columns ===")
print(df.columns.tolist())

# Basic statistics for categorical columns
print("\n=== Categorical Columns Summary ===")
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nCounts for {col}:")
    print(df[col].value_counts())
    print(f"\nPercentages for {col}:")
    print(df[col].value_counts(normalize=True) * 100)

# Save summary to file (optional)
with open('eda_summary.txt', 'w') as f:
    f.write("=== Dataset Info ===\n")
    df.info(buf=f)
    f.write("\n=== Summary Statistics ===\n")
    df.describe().to_string(buf=f)