"""
Feature engineering για το μοντέλο πρόβλεψης S&P 500.

Όλα τα features είναι αδιάστατα (ποσοστά / λόγοι / λογάριθμοι), ώστε να μην
εξαρτώνται από την απόλυτη κλίμακα τιμής του SPY.

Ταυτόσημη λογική με το επικυρωμένο Kaggle pipeline (§3), για να παραμένουν
τα αποτελέσματα αναπαραγώγιμα.
"""

import numpy as np
import pandas as pd

from data_collection import load_snapshot

# Τα δύο σύνολα χαρακτηριστικών που συγκρίναμε στην έρευνα
FEATURES_TECH = [
    "LogReturn", "LogVolume", "SMA_10_rel", "SMA_50_rel", "SMA_200_rel",
    "RSI_norm", "Volatility_20",
]
FEATURES_MACRO = ["LogVIX", "YieldSpread"]
TARGET = "LogReturn"


def compute_rsi(series, period=14):
    """Relative Strength Index, κλασικός τύπος 14 ημερών."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    """
    Παίρνει το ωμό snapshot (OHLCV + VIX + yields) και επιστρέφει το
    DataFrame με όλα τα τεχνικά και macro features υπολογισμένα.

    Δεν κάνει dropna εδώ σκόπιμα — αυτό γίνεται στο στάδιο του split,
    ώστε ο καθένας που καλεί τη συνάρτηση να βλέπει πόσες γραμμές NaN
    δημιουργεί κάθε feature.
    """
    df = market.copy().sort_values("Date").reset_index(drop=True)

    # --- στόχος & βασικά ---
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["LogVolume"] = np.log(df["Volume"].replace(0, np.nan))

    # --- τεχνικοί δείκτες ---
    for window in (10, 50, 200):
        sma = df["Close"].rolling(window=window).mean()
        df[f"SMA_{window}_rel"] = (df["Close"] - sma) / df["Close"]

    df["RSI_14"] = compute_rsi(df["Close"], period=14)
    df["RSI_norm"] = (df["RSI_14"] - 50) / 50
    df["Volatility_20"] = df["LogReturn"].rolling(window=20).std()

    # --- macro ---
    df["LogVIX"] = np.log(df["VIX"])
    df["YieldSpread"] = df["Yield10Y"] - df["Yield3M"]
    df["RiskFreeDaily"] = (df["Yield3M"] / 100.0) / 252.0  # για backtest

    return df


if __name__ == "__main__":
    market = load_snapshot()
    data = build_features(market)

    print(f"Γραμμές: {len(data)}")
    print(f"Τεχνικά features: {FEATURES_TECH}")
    print(f"Macro features: {FEATURES_MACRO}")
    print("\nΤελευταίες γραμμές:")
    cols = ["Date", "Close", "LogReturn", "RSI_norm", "LogVIX", "YieldSpread"]
    print(data[cols].tail(3))

    print("\nNaN ανά νέα στήλη (αναμενόμενο στην αρχή λόγω rolling windows):")
    new_cols = FEATURES_TECH + FEATURES_MACRO
    print(data[new_cols].isna().sum())