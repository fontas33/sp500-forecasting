"""
Συλλογή δεδομένων αγοράς για την πρόβλεψη του S&P 500.

Κατεβάζει OHLCV του SPY μαζί με macro δείκτες (VIX, αποδόσεις 10ετούς/3μήνου)
και τα αποθηκεύει ως παγωμένο snapshot, ώστε τα πειράματα να είναι αναπαραγώγιμα.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "market_snapshot.csv"
META_PATH = DATA_DIR / "snapshot_meta.json"

TICKERS = {
    "SPY": None,
    "^VIX": "VIX",
    "^TNX": "Yield10Y",
    "^IRX": "Yield3M",
}

def download_with_retry(ticker, col_name, start="1993-01-01",
                        max_retries=6, min_rows=4000, pause=4):
    """
    Κατεβάζει ιστορικά δεδομένα με επαναλήψεις.

    Το Yahoo Finance επιστρέφει περιστασιακά ελλιπή δεδομένα (π.χ. 16 γραμμές
    αντί για 8000) χωρίς να σηκώνει σφάλμα. Ελέγχουμε το πλήθος γραμμών και
    ξαναδοκιμάζουμε αν είναι προφανώς ελλιπές.
    """
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()

            if ticker == "SPY":
                out = raw[["Date", "Open", "High", "Low", "Close", "Volume"]]
            else:
                out = raw[["Date", "Close"]].rename(columns={"Close": col_name})

            if len(out) >= min_rows:
                print(f"  {ticker}: OK ({len(out)} γραμμές, προσπάθεια {attempt})")
                return out

            print(f"  {ticker}: προσπάθεια {attempt} -> {len(out)} γραμμές, retry...")
        except Exception as exc:
            print(f"  {ticker}: προσπάθεια {attempt} απέτυχε ({type(exc).__name__}), retry...")

        time.sleep(pause)

    raise RuntimeError(f"{ticker}: αποτυχία μετά από {max_retries} προσπάθειες")


def build_snapshot(force=False):
    """Κατεβάζει και αποθηκεύει το snapshot. Αν υπάρχει ήδη, το φορτώνει."""
    DATA_DIR.mkdir(exist_ok=True)

    if SNAPSHOT_PATH.exists() and not force:
        meta = json.loads(META_PATH.read_text())
        print(f"Snapshot υπάρχει ήδη (ελήφθη: {meta['downloaded_at']})")
        return load_snapshot()

    print("Λήψη δεδομένων αγοράς...")
    frames = {t: download_with_retry(t, c) for t, c in TICKERS.items()}

    market = frames["SPY"]
    for ticker in ("^VIX", "^TNX", "^IRX"):
        market = market.merge(frames[ticker], on="Date", how="left")

    market.to_csv(SNAPSHOT_PATH, index=False)
    META_PATH.write_text(json.dumps({
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(market),
        "tickers": list(TICKERS),
        "date_range": [str(market["Date"].min().date()),
                       str(market["Date"].max().date())],
    }, indent=2))

    print(f"Snapshot αποθηκεύτηκε: {len(market)} γραμμές -> {SNAPSHOT_PATH}")
    return market


def load_snapshot():
    """Φορτώνει το αποθηκευμένο snapshot."""
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε snapshot στο {SNAPSHOT_PATH}. "
            "Τρέξε: python src/data_collection.py"
        )
    return pd.read_csv(SNAPSHOT_PATH, parse_dates=["Date"])


if __name__ == "__main__":
    market = build_snapshot()
    print(f"\nΕύρος: {market['Date'].min().date()} -> {market['Date'].max().date()}")
    print(market.tail(3))