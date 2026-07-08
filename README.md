# Women's E-Commerce Clothing Reviews - NLP Sentiment Analysis

## Project Overview

This project performs sentiment analysis on customer reviews from the Women's E-Commerce Clothing Reviews dataset.

The reviews are cleaned using common Natural Language Processing (NLP) preprocessing techniques and analyzed with the VADER Sentiment Analyzer to classify each review as **Positive**, **Negative**, or **Neutral**.

The project also includes several visualizations to better understand customer opinions and sentiment patterns.

---

## Dataset

- Source: Kaggle
- Dataset: Women's E-Commerce Clothing Reviews
- Language: English
- Format: CSV

---

## Technologies Used

- Python
- Pandas
- NLTK
- VADER Sentiment Analyzer
- Matplotlib
- Collections (Counter)
- Regular Expressions (re)

---

## Features

- Load customer reviews from CSV
- Remove missing reviews
- Clean review texts
  - Convert to lowercase
  - Remove punctuation
  - Remove numbers
  - Remove extra spaces
  - Remove stop words
- Apply VADER Sentiment Analysis
- Calculate compound sentiment scores
- Label reviews as:
  - Positive
  - Neutral
  - Negative
- Export processed dataset
- Generate visualizations

---

## Visualizations

The project creates the following charts:

- Average Compound Score by Rating
- Top 15 Positive Words
- Top 15 Negative Words
- Average VADER Emotion Scores

---

## Project Structure

```
.
├── dataset/
│   └── Womens Clothing E-Commerce Reviews.csv
├── outputs/
│   ├── average_compound_score.png
│   ├── top_positive_words.png
│   ├── top_negative_words.png
│   └── vader_scores.png
├── sentiment_results.csv
├── main.py
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/yourusername/i2i-Academy-NLP-11.git
```

```bash
cd i2i-Academy-NLP-11
```

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Output

The program:

- Cleans review texts
- Performs sentiment analysis using VADER
- Saves the processed dataset as `sentiment_results.csv`
- Generates visualization charts inside the `outputs` folder

---

## Learning Outcomes

Through this project I learned:

- Basic NLP preprocessing
- Text cleaning techniques
- Stop words removal
- Sentiment analysis using VADER
- Customer review analysis
- Data visualization with Matplotlib

---
