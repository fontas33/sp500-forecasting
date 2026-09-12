"""
Κατασκευή features σε πραγματικό χρόνο για live πρόβλεψη.

Kάθε βήμα εδώ ΠΡΈΠΕΙ να αντιγράφει ακριβώς το training pipeline
(features.py + alpaca_sentiment.py). Οποιαδήποτε απόκλιση παράγει σιωπηλά
λάθος προβλέψεις.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "trading"))

from features import build_features, FEATURES_TECH
from alpaca_client import get_news, AlpacaError

MODEL_DIR = ROOT / "models"
DEFAULT_FEATURE_SET = "full10_alpaca"

# Πόσες ημερολογιακές ημέρες πίσω κατεβάζουμε — αρκετές ώστε να καλύπτουν
# το SMA_200 (200 trading days ≈ 290 ημερολογιακές) συν το lookback.
HISTORY_DAYS = 700


class LiveFeatureError(Exception):
    """Αδυναμία κατασκευής έγκυρων features."""


def load_metadata(feature_set=DEFAULT_FEATURE_SET):
    path = MODEL_DIR / feature_set / "production_metadata.json"
    if not path.exists():
        raise LiveFeatureError(
            f"Δεν βρέθηκε {path}. Τρέξε πρώτα:\n" 
            f" python src/persist_model.py --feature-set {feature_set}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_market_data():
    """Πρόσφατα δεδομένα αγοράς — ίδιοι tickers με το snapshot."""
    start = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")

    frames = {}
    for ticker, col in [("SPY", None), ("^VIX", "VIX"),
                        ("^TNX", "Yield10Y"), ("^IRX", "Yield3M")]:
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        if ticker == "SPY":
            frames[ticker] = raw[["Date", "Open", "High", "Low", "Close", "Volume"]]
        else:
            frames[ticker] = raw[["Date", "Close"]].rename(columns={"Close": col})

    market = frames["SPY"]
    for t in ("^VIX", "^TNX", "^IRX"):
        market = market.merge(frames[t], on="Date", how="left")

    if len(market) < 250:
        raise LiveFeatureError(
            f"Ανεπαρκή δεδομένα αγοράς: {len(market)} γραμμές (χρειάζονται >250 "
            "για το SMA_200)"
        )
    return market


def fetch_recent_sentiment(days_back=30):
    """
    Sentiment των τελευταίων ημερών από το Alpaca news API.

    Χρησιμοποιεί το ΙΔΙΟ μοντέλο (FinBERT) και την ίδια λογική συγκέντρωσης
    με το score_alpaca_sentiment.py.
    """
    from transformers import pipeline

    start = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    rows, token = [], None

    for _ in range(100):
        kwargs = {"symbols": "SPY,SPX,QQQ,DIA", "start": start, "limit": 50}
        if token:
            kwargs["page_token"] = token
        r = get_news(**kwargs)
        for item in r.get("news", []):
            rows.append({"created_at": item.get("created_at"),
                         "headline": item.get("headline", "")})
        token = r.get("next_page_token")
        if not token:
            break

    if not rows:
        return pd.DataFrame(columns=["Date", "SentimentMean", "SentimentCount"])

    news = pd.DataFrame(rows)
    news["created_at"] = pd.to_datetime(news["created_at"], utc=True, errors="coerce")
    news = news.dropna(subset=["created_at"])

    device_id = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                    device=device_id, batch_size=32)

    texts = news["headline"].fillna("").astype(str).tolist()
    labels, scores = [], []
    for i in range(0, len(texts), 32):
        for res in pipe(texts[i:i + 32], truncation=True, max_length=128):
            labels.append(res["label"])
            scores.append(res["score"])

    sign = pd.Series(labels).map({"positive": 1.0, "negative": -1.0, "neutral": 0.0})
    news["num"] = (sign * pd.Series(scores)).values

    # Ίδια μετατροπή σε US/Eastern όπως στο training
    news["trade_date"] = news["created_at"].dt.tz_convert("America/New_York").dt.date

    daily = (news.groupby("trade_date")
             .agg(SentimentMean=("num", "mean"), SentimentCount=("num", "size"))
             .reset_index().rename(columns={"trade_date": "Date"}))
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily


def build_live_window(feature_set=DEFAULT_FEATURE_SET):
    """
    Επιστρέφει (X, info) όπου X έχει σχήμα (1, lookback, n_features),
    έτοιμο για το μοντέλο.
    """
    meta = load_metadata(feature_set)
    features, lookback = meta["features"], meta["lookback"]
    needs_sentiment = "SentimentMean" in features

    market = fetch_market_data()
    data = build_features(market).dropna(subset=FEATURES_TECH).reset_index(drop=True)

    sentiment_info = {}
    if needs_sentiment:
        daily = fetch_recent_sentiment(days_back=max(90, lookback * 2))
        if daily.empty:
            raise LiveFeatureError("Καμία είδηση τις τελευταίες 30 ημέρες — "
                                   "πιθανό πρόβλημα στο news API")

        data = data.merge(daily, on="Date", how="left")

        last_real = data.loc[data["SentimentMean"].notna(), "Date"].max()
        staleness = (data["Date"].max() - last_real).days
        max_stale = meta.get("max_sentiment_staleness_days", 5)

        if staleness > max_stale:
            raise LiveFeatureError(
                f"Το sentiment είναι {staleness} ημερών (όριο: {max_stale}). "
                "Το feature θεωρείται μπαγιάτικο — καμία πρόβλεψη."
            )

        n_filled = data["SentimentMean"].isna().sum()
        data["SentimentMean"] = data["SentimentMean"].ffill()
        data["SentimentCount"] = data["SentimentCount"].fillna(0)
        sentiment_info = {"staleness_days": int(staleness),
                          "ffilled_rows": int(n_filled),
                          "last_real_sentiment": str(last_real.date())}

    data = data.dropna(subset=features).reset_index(drop=True)
    if len(data) < lookback:
        raise LiveFeatureError(f"Μόνο {len(data)} έγκυρες γραμμές, χρειάζονται {lookback}")

    window = data[features].values[-lookback:]
    X = window.reshape(1, lookback, len(features))

    info = {
        "as_of_date": str(data["Date"].iloc[-1].date()),
        "last_close": float(data["Close"].iloc[-1]),
        "rows_available": len(data),
        "feature_set": meta["feature_set"],
        **sentiment_info,
    }
    return X, info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    args = ap.parse_args()

    X, info = build_live_window(args.feature_set)
    print("Live feature window κατασκευάστηκε\n")
    for k, v in info.items():
        print(f"  {k:24s}: {v}")
    print(f"\n  shape                   : {X.shape}")
    print(f"  τιμές (min/max)         : {X.min():.4f} / {X.max():.4f}")