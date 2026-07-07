import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Womens Clothing E-Commerce Reviews.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset shape before cleaning
print("\nDataset Shape Before Cleaning:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Remove rows with missing review text
df = df.dropna(subset=["Review Text"])

# Dataset shape after cleaning
print("\nDataset Shape After Cleaning:")
print(df.shape)

# Display first review texts
print("\nSample Reviews:")
print(df["Review Text"].head())

# Rating distribution
print("\nRating Distribution:")
print(df["Rating"].value_counts())