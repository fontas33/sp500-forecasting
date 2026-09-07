"""
FinBERT sentiment scoring στις ιστορικές ειδήσεις του Alpaca.

Παράγει ΔΥΟ εκδοχές:
  - headline μόνο -> άμεσα συγκρίσιμο με το Kaggle dataset
  - headline + summary -> πλουσιότερο κείμενο, ανοιχτό ερώτημα το άν βοηθάει
  
Checkpointing ανά batch: αν διακοπεί, συνεχίζει από εκεί που σταμάτησεε.
"""

from pathlib import Path

import numpy as np 
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "alpaca_news_raw.csv"

BATCH_SIZE = 64 
MAX_LENGTH = 128       # μεγαλύτερο από τα 64 του  Kaggle, για να χωρά το summary
CHECKPOINT_EVERY = 50  # batches 

VARIANTS = {
    "headline": lambda df: df["headline"].fillna("").astype(str),
    "headline_summary": lambda df:(
        df["headline"].fillna("").astype(str) + ". "+ df["summary"].fillna("").astype(str)
    ).str.strip(". "),
}

def score_variant(texts, variant_name, pipe):
    """Τρέχει FinBERT σε λίστα κειμένων, με checkpointing."""
    ckpt = DATA_DIR / f"alpaca_scores_{variant_name}_partial.csv"

    start_idx = 0
    labels, scores = [], []
    if ckpt.exists():
        done = pd.read_csv(ckpt)
        labels = done["label"].tolist()
        scores = done["score"].tolist()
        start_idx = len(labels)
        print(f"  Συνέχεια από {start_idx}/{len(texts)}")

    for i in tqdm(range(start_idx, len(texts), BATCH_SIZE),
                  desc=f"FinBERT [{variant_name}]",
                  initial=start_idx // BATCH_SIZE,
                  total=(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for r in pipe(batch, truncation=True, max_length=MAX_LENGTH):
            labels.append(r["label"])
            scores.append(r["score"])

        if (i // BATCH_SIZE) % CHECKPOINT_EVERY == 0:
            pd.DataFrame({"label": labels, "score": scores}).to_csv(ckpt, index=False)

    pd.DataFrame({"label": labels, "score": scores}).to_csv(ckpt, index=False)
    return labels, scores


def to_numeric(labels, scores):
    sign = pd.Series(labels).map({"positive": 1.0, "negative": -1.0, "neutral": 0.0})
    return (sign * pd.Series(scores)).values


def main():
    news = pd.read_csv(RAW_PATH, parse_dates=["created_at"])
    news = news.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    print(f"{len(news)} άρθρα, {news['created_at'].min().date()} -> "
          f"{news['created_at'].max().date()}\n")

    device_id = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'GPU' if device_id == 0 else 'CPU'}")
    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                    device=device_id, batch_size=BATCH_SIZE)

    for variant_name, extractor in VARIANTS.items():
        print(f"\n--- Variant: {variant_name} ---")
        texts = extractor(news).tolist()
        labels, scores = score_variant(texts, variant_name, pipe)
        news[f"sent_{variant_name}"] = to_numeric(labels, scores)

    # --- ημερήσια συγκέντρωση ---
    # Ημερομηνία σε US/Eastern: μια είδηση στις 22:00 ET ανήκει σε εκείνη
    # την ημέρα συναλλαγών, όχι στην επόμενη (που θα έδινε το UTC).
    news["trade_date"] = (news["created_at"].dt.tz_convert("America/New_York")
                          .dt.date)

    daily = (news.groupby("trade_date")
             .agg(SentimentMean=("sent_headline", "mean"),
                  SentimentMeanFull=("sent_headline_summary", "mean"),
                  SentimentCount=("sent_headline", "size"))
             .reset_index()
             .rename(columns={"trade_date": "Date"}))
    daily["Date"] = pd.to_datetime(daily["Date"])

    out = DATA_DIR / "alpaca_sentiment_daily.csv"
    daily.to_csv(out, index=False)

    print(f"\n{len(daily)} ημέρες με sentiment")
    print(daily[["SentimentMean", "SentimentMeanFull", "SentimentCount"]].describe())

    corr = daily["SentimentMean"].corr(daily["SentimentMeanFull"])
    print(f"\nΣυσχέτιση headline vs headline+summary: {corr:.4f}")
    print(f"Αποθηκεύτηκε: {out}")


if __name__ == "__main__":
    main()
