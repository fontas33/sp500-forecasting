"""
Φόρτωση προϋπολογισμένου sentiment από χρηματοοικονομικές ειδήσεις.

Τα FinBERT scores υπολογίστηκαν στο Kaggle (χρειάζεται GPU + το πρωτότυπο
dataset ειδήσεων) και είναι ήδη αποθηκευμένα εδώ. Αυτό το module απλά τα
συγκεντρώνει σε ημερήσιο επίπεδο και τα ενώνει με τα υπόλοιπα features.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SENTIMENT_CACHE = DATA_DIR / "news_sentiment_scored.csv"


def load_daily_sentiment():
    """
    Επιστρέφει DataFrame με στήλες: Date, SentimentMean, SentimentCount.

    SentimentMean: μέσος όρος (θετικό score αν positive, αρνητικό αν
    negative, 0 αν neutral) όλων των τίτλων της ημέρας.
    """
    if not SENTIMENT_CACHE.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε {SENTIMENT_CACHE}. Χρειάζεται το cached αρχείο από "
            "το Kaggle commit (FinBERT scoring απαιτεί GPU inference σε "
            "19k+ τίτλους — τρέχει εκεί, όχι εδώ)."
        )

    news = pd.read_csv(SENTIMENT_CACHE, parse_dates=["Date"])

    sign = news["Sentiment"].map({"positive": 1.0, "negative": -1.0, "neutral": 0.0})
    news["SentimentNumeric"] = sign * news["SentimentScore"]

    daily = (news.groupby("Date")
             .agg(SentimentMean=("SentimentNumeric", "mean"),
                  SentimentCount=("SentimentNumeric", "size"))
             .reset_index())
    return daily


def attach_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ενώνει το sentiment με το feature DataFrame, περιορίζει στο εύρος
    κάλυψης ειδήσεων, και κάνει forward-fill στις ημέρες χωρίς είδηση.

    Σημείωση περιορισμού (MNAR, βλ. έρευνα §5): οι ημέρες χωρίς κάλυψη
    είναι συστηματικά πιο ασταθείς — το ffill εισάγει μικρή μεροληψία.
    """
    daily = load_daily_sentiment()

    merged = data.merge(daily, on="Date", how="left")
    merged = merged[(merged["Date"] >= daily["Date"].min()) &
                    (merged["Date"] <= daily["Date"].max())].reset_index(drop=True)

    merged["SentimentMean"] = merged["SentimentMean"].ffill()
    merged["SentimentCount"] = merged["SentimentCount"].fillna(0)

    return merged


if __name__ == "__main__":
    from data_collection import load_snapshot
    from features import build_features, FEATURES_TECH, TARGET

    market = load_snapshot()
    data = build_features(market).dropna(subset=FEATURES_TECH + [TARGET]).reset_index(drop=True)
    with_sentiment = attach_sentiment(data)

    print(f"Πριν: {len(data)} γραμμές (πλήρες ιστορικό)")
    print(f"Μετά: {len(with_sentiment)} γραμμές (περιορισμένο σε κάλυψη ειδήσεων)")
    print(f"Εύρος: {with_sentiment['Date'].min().date()} -> {with_sentiment['Date'].max().date()}")
    print(f"NaN σε SentimentMean: {with_sentiment['SentimentMean'].isna().sum()}")