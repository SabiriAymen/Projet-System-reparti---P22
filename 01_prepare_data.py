"""
Etape 1 : Préparation des données
- Charge les prix journaliers AAPL (2012-2017)
- Charge les tweets AAPL (2014-2016), calcule un score de sentiment VADER par tweet
- Agrège le sentiment par jour (moyenne + volume de tweets)
- Fusionne prix + sentiment -> dataset final propre
"""
import json
import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

BASE = "data/stocknet-dataset-master"
PRICE_PATH = f"{BASE}/price/raw/AAPL.csv"
TWEET_DIR = f"{BASE}/tweet/preprocessed/AAPL"

# ---- 1. Prix ----
price = pd.read_csv(PRICE_PATH, parse_dates=["Date"])
price = price.sort_values("Date").reset_index(drop=True)
print(f"Prix chargés : {len(price)} lignes, du {price.Date.min().date()} au {price.Date.max().date()}")

# ---- 2. Sentiment des tweets (VADER) ----
sia = SentimentIntensityAnalyzer()
rows = []
for fname in sorted(os.listdir(TWEET_DIR)):
    fpath = os.path.join(TWEET_DIR, fname)
    daily_scores = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tweet = json.loads(line)
            text = " ".join(tweet["text"])
            score = sia.polarity_scores(text)["compound"]
            daily_scores.append(score)
    if daily_scores:
        rows.append({
            "Date": pd.to_datetime(fname),
            "sentiment_mean": sum(daily_scores) / len(daily_scores),
            "tweet_volume": len(daily_scores),
        })

sentiment = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
print(f"Sentiment calculé pour {len(sentiment)} jours (moyenne VADER par jour)")

# ---- 3. Fusion ----
df = price.merge(sentiment, on="Date", how="left")
# Les jours sans tweet -> sentiment neutre (0) et volume 0
df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)
df["tweet_volume"] = df["tweet_volume"].fillna(0).astype(int)

# On restreint à la période où le sentiment est disponible (2014-2016)
df = df[(df["Date"] >= sentiment.Date.min()) & (df["Date"] <= sentiment.Date.max())].reset_index(drop=True)

os.makedirs("output", exist_ok=True)
df.to_csv("output/dataset_final.csv", index=False)
print(f"\nDataset final : {len(df)} lignes -> output/dataset_final.csv")
print(df.head())
print("\nStatistiques sentiment:")
print(df["sentiment_mean"].describe())
