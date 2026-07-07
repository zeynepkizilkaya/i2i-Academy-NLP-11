import re
import matplotlib.pyplot as plt
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
df[["Compound Score", "Sentiment"]] = df["Review Text"].apply(get_sentiment)

print("\nSentiment Distribution:\n")
print(df["Sentiment"].value_counts())

print("\nAverage Compound Score:")
print(df["Compound Score"].mean())

print("\nTop 5 Positive Reviews:\n")
print(df.nlargest(5, "Compound Score")[["Review Text", "Compound Score"]])

print("\nTop 5 Negative Reviews:\n")
print(df.nsmallest(5, "Compound Score")[["Review Text", "Compound Score"]])
# Save processed dataset
df.to_csv("sentiment_results.csv", index=False)

print("\nProcessed dataset saved successfully.")

# -----------------------------
# Sentiment Distribution Chart
# -----------------------------

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(6,4))

plt.bar(sentiment_counts.index, sentiment_counts.values)

plt.title("Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("outputs/sentiment_distribution.png")

plt.show()


# -----------------------------
# Rating vs Sentiment
# -----------------------------

rating_sentiment = pd.crosstab(df["Rating"], df["Sentiment"])

rating_sentiment.plot(kind="bar", figsize=(8,5))

plt.title("Rating vs Sentiment")

plt.xlabel("Rating")

plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("outputs/rating_vs_sentiment.png")

plt.show()