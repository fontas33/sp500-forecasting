"""Σύγκριση των δύο πηγών sentiment: Kaggle dataset vs Alpaca news API."""

from pathlib import Path

import numpy as np
import pandas as pd 
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# --- Kaggle sentiment (ίδια λογική με το sentiment.py) ---
kaggle = pd.read_csv(DATA_DIR / "news_sentiment_scored.csv", parse_dates=["Date"])
sign = kaggle["Sentiment"].map({"positive": 1.0, "negative": -1.0, "neutral": 0.0})
kaggle["num"] = sign * kaggle["SentimentScore"]
kaggle_daily = (kaggle.groupby("Date")
                .agg(kaggle_mean=("num", "mean"), kaggle_count=("num", "size"))
                .reset_index())

# --- Alpaca sentiemnt ---
alpaca = pd.read_csv(DATA_DIR / "alpaca_sentiment_daily.csv", parse_dates=["Date"])

merged = kaggle_daily.merge(alpaca, on="Date", how="inner")
print(f"Κοινές ημέρες: {len(merged)}")
print(f"Εύρος επικάλυψης: {merged['Date'].min().date()} -> {merged['Date'].max().date()}\n")

print("=== Περιγραφικά ===")
print(f"{'':22s} {'Kaggle':>10s} {'Alpaca':>10s}")
for label, a, b in [
    ("mean", merged["kaggle_mean"].mean(), merged["SentimentMean"].mean()),
    ("std", merged["kaggle_mean"].std(), merged["SentimentMean"].std()),
    ("median", merged["kaggle_mean"].median(), merged["SentimentMean"].median()),
    ("άρθρα/ημέρα (median)", merged["kaggle_count"].median(), merged["SentimentCount"].median()),
]:
    print(f"{label:22s} {a:10.4f} {b:10.4f}")

r_p, p_p = stats.pearsonr(merged["kaggle_mean"], merged["SentimentMean"])
r_s, p_s = stats.spearmanr(merged["kaggle_mean"], merged["SentimentMean"])
ks_stat, ks_p = stats.ks_2samp(merged["kaggle_mean"], merged["SentimentMean"])

print(f"\n=== Συμφωνία των δύο πηγών ===")
print(f"Pearson r  : {r_p:.4f} (p={p_p:.3g})")
print(f"Spearman ρ : {r_s:.4f} (p={p_s:.3g})")
print(f"KS test    : D={ks_stat:.4f}, p={ks_p:.3g} "
      f"-> {'ΔΙΑΦΟΡΕΤΙΚΕΣ κατανομές' if ks_p < 0.05 else 'συμβατές κατανομές'}")

agree = ((merged["kaggle_mean"] > 0) == (merged["SentimentMean"] > 0)).mean()
print(f"\nΣυμφωνία προσήμου: {agree*100:.1f}% των ημερών")