"""
Φόρτωση και προσάρτηση του Alpaca sentiment (live-συμβατή πηγή).

Σε αντίθεση με το sentiment.py (Kaggle dataset, ιστορικό μόνο), αυτή η πηγή
τροφοδοτείται από το Alpaca news API και είναι διαθέσιμη σε πραγματικό χρόνο.

Οι δύο πηγές ΔΕΝ είναι εναλλάξιμες: Pearson r = 0.074 στις κοινές ημέρες.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SENTIMENT_PATH = DATA_DIR / "alpaca_sentiment_daily.csv"


def load_alpaca_sentiment():
    if not SENTIMENT_PATH.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε {SENTIMENT_PATH}. Τρέξε πρώτα:\n"
            "  python src/alpaca_news.py\n"
            "  python src/score_alpaca_sentiment.py"
        )
    return pd.read_csv(SENTIMENT_PATH, parse_dates=["Date"])


def attach_alpaca_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ενώνει το Alpaca sentiment, περιορίζει στο εύρος κάλυψης, ffill στα κενά.

    Το ffill είναι χρονικά νόμιμο (παρελθόν -> μέλλον). Η κάλυψη εδώ είναι
    πρακτικά πλήρης στα trading days, οπότε τα κενά είναι ελάχιστα — σε
    αντίθεση με το Kaggle dataset (13.8% κενά, MNAR).
    """
    daily = load_alpaca_sentiment()

    merged = data.merge(daily, on="Date", how="left")
    merged = merged[(merged["Date"] >= daily["Date"].min()) &
                    (merged["Date"] <= daily["Date"].max())].reset_index(drop=True)

    n_missing = merged["SentimentMean"].isna().sum()
    merged["SentimentMean"] = merged["SentimentMean"].ffill()
    merged["SentimentMeanFull"] = merged["SentimentMeanFull"].ffill()
    merged["SentimentCount"] = merged["SentimentCount"].fillna(0)

    if n_missing:
        pct = n_missing / len(merged) * 100
        print(f"  [alpaca_sentiment] {n_missing} κενές ημέρες ({pct:.1f}%) -> ffill")

    return merged


if __name__ == "__main__":
    from data_collection import load_snapshot
    from features import build_features, FEATURES_TECH, TARGET

    market = load_snapshot()
    data = build_features(market).dropna(subset=FEATURES_TECH + [TARGET]).reset_index(drop=True)
    result = attach_alpaca_sentiment(data)

    print(f"Πριν: {len(data)} γραμμές")
    print(f"Μετά: {len(result)} γραμμές")
    print(f"Εύρος: {result['Date'].min().date()} -> {result['Date'].max().date()}")
    print(f"NaN: {result[['SentimentMean', 'SentimentMeanFull']].isna().sum().to_dict()}")
    print(f"\nSentimentCount: median={result['SentimentCount'].median():.0f}, "
          f"min={result['SentimentCount'].min():.0f}, max={result['SentimentCount'].max():.0f}")