import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download stopwords
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("dataset/Womens Clothing E-Commerce Reviews.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape Before Cleaning:")
print(df.shape)

# Remove rows without review
df = df.dropna(subset=["Review Text"])

print("\nDataset Shape After Cleaning:")
print(df.shape)

# Stop words
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Clean reviews
df["Clean Review"] = df["Review Text"].apply(clean_text)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return pd.Series([score, label])


# Analyze sentiment
df[["Compound Score", "Sentiment"]] = df["Clean Review"].apply(get_sentiment)

print("\nSentiment Distribution:\n")
print(df["Sentiment"].value_counts())

print("\nAverage Compound Score:")
print(df["Compound Score"].mean())

print("\nTop 5 Positive Reviews:\n")
print(df.nlargest(5, "Compound Score")[["Review Text", "Compound Score"]])

print("\nTop 5 Negative Reviews:\n")
print(df.nsmallest(5, "Compound Score")[["Review Text", "Compound Score"]])