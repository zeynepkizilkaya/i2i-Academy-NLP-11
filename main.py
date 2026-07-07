import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
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

custom_stopwords = {
    "dress", "top", "shirt", "wear", "wearing",
    "size", "fabric", "product", "item",
    "one", "would", "im", "look", "color",
    "like", "really", "ordered", "order",
    "back", "also", "well", "even", "fit"
}

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
# Average Compound Score by Rating
# -----------------------------

average_scores = (
    df.groupby("Rating")["Compound Score"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8,5))

plt.plot(
    average_scores["Rating"],
    average_scores["Compound Score"],
    marker="o",
    linewidth=3
    
)

for x, y in zip(
        average_scores["Rating"],
        average_scores["Compound Score"]):
    plt.text(
        x,
        y + 0.02,
        f"{y:.2f}",
        ha="center",
        fontsize=10
    )

plt.grid(alpha=0.3)

plt.title(
    "Average Compound Score by Rating",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Rating")

plt.ylabel("Average Compound Score")

plt.savefig(
    "outputs/average_compound_score.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Top Positive Words
# -----------------------------

positive_reviews = " ".join(
    df[df["Sentiment"]=="Positive"]["Clean Review"]
)

positive_counter = Counter(
    positive_reviews.split()
)

positive_words = positive_counter.most_common(15)

words = [w for w, c in positive_words]
counts = [c for w, c in positive_words]

plt.figure(figsize=(10,6))

# Positive
plt.barh(words, counts, color="forestgreen")

plt.title(
    "Top 15 Positive Words",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Frequency")

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/top_positive_words.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Top Negative Words
# -----------------------------

negative_reviews = " ".join(
    df[df["Sentiment"]=="Negative"]["Clean Review"]
)

negative_counter = Counter(
    negative_reviews.split()
)

negative_words = negative_counter.most_common(15)

words = [w for w, c in negative_words]
counts = [c for w, c in negative_words]

plt.figure(figsize=(10,6))

# Negative
plt.barh(words, counts, color="firebrick")

plt.title(
    "Top 15 Negative Words",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Frequency")

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/top_negative_words.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Average Emotion Scores
# -----------------------------

df["Positive Score"] = df["Review Text"].apply(
    lambda x: analyzer.polarity_scores(x)["pos"]
)

df["Neutral Score"] = df["Review Text"].apply(
    lambda x: analyzer.polarity_scores(x)["neu"]
)

df["Negative Score"] = df["Review Text"].apply(
    lambda x: analyzer.polarity_scores(x)["neg"]
)

scores = [
    df["Positive Score"].mean(),
    df["Neutral Score"].mean(),
    df["Negative Score"].mean()
]

labels = [
    "Positive",
    "Neutral",
    "Negative"
]

plt.figure(figsize=(7,5))

bars = plt.bar(labels, scores)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x()+bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center"
    )

plt.title(
    "Average VADER Emotion Scores",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("Average Score")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/vader_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

