import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download stopwords (only runs first time)
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("dataset/Womens Clothing E-Commerce Reviews.csv")

# Display dataset information
print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape Before Cleaning:")
print(df.shape)

# Remove rows with missing reviews
df = df.dropna(subset=["Review Text"])

print("\nDataset Shape After Cleaning:")
print(df.shape)

# English stop words
stop_words = set(stopwords.words("english"))


def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Apply cleaning function
df["Clean Review"] = df["Review Text"].apply(clean_text)

print("\nOriginal Review:\n")
print(df["Review Text"].iloc[0])

print("\nCleaned Review:\n")
print(df["Clean Review"].iloc[0])