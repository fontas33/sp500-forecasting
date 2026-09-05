"""
Λήψη ιστορικών ειδήσεων από το Alpaca news API.

Κατεβάζει σε μηνιαία κομμάτια με pagination και checkpointing: αν διακοπεί,
συνεχίζει από εκεί που σταμάτησε αντί να ξεκινήσει από την αρχή.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "trading"))

from alpaca_client import get_news, AlpacaError

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "alpaca_news_raw.csv"
PROGRESS_PATH = DATA_DIR / "alpaca_news_progress.json"

SYMBOLS = "SPY,SPX,QQQ,DIA" # δείκτες-proxy για ευρεία αγορά
START_YEAR = 2015           # ξεκινάμε αισιόδοξα, το API θα δείξει πού σταματά
PAGE_LIMIT = 50
PAUSE = 0.35                # ~3 req/sec, συντηρητικό για rate limits

def month_range(start_year, end_year=None):
    end_year = end_year or date.today().year
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if date(year, month, 1) > date.today():
                return
            yield year, month


def fetch_month(year, month, max_pages=200):
    """Κατεβάζει όλα τα άρθρα ενός μήνα, ακολουθώντας το pagination."""
    start = f"{year}-{month:02d}-01"
    end_month = month % 12 + 1
    end_year = year + (1 if month == 12 else 0)
    end = f"{end_year}-{end_month:02d}-01"

    rows, token = [], None
    for _ in range(max_pages):
        kwargs = {"symbols": SYMBOLS, "start": start, "end": end, "limit": PAGE_LIMIT}
        if token:
            kwargs["page_token"] = token

        for attempt in range(4):
            try:
                r = get_news(**kwargs)
                break
            except AlpacaError as exc:
                if "429" in str(exc):
                    wait = 3 * (attempt + 1)
                    print(f"    rate limit, αναμονή {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise AlpacaError(f"Αποτυχία μετά από 4 προσπάθειες: {year}-{month:02d}")

        for item in r.get("news", []):
            rows.append({
                "created_at": item.get("created_at"),
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "symbols": ",".join(item.get("symbols", [])),
            })

        token = r.get("next_page_token")
        if not token:
            break
        time.sleep(PAUSE)

    return rows


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"completed": [], "empty_streak": 0}


def save_progress(progress):
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def download_all():
    DATA_DIR.mkdir(exist_ok=True)
    progress = load_progress()
    completed = set(progress["completed"])

    existing = pd.read_csv(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame()
    all_rows = existing.to_dict("records") if len(existing) else []
    print(f"Ξεκινώ. Ήδη αποθηκευμένα: {len(all_rows)} άρθρα, "
          f"{len(completed)} μήνες ολοκληρωμένοι\n")

    empty_streak = progress["empty_streak"]

    for year, month in month_range(START_YEAR):
        key = f"{year}-{month:02d}"
        if key in completed:
            continue

        try:
            rows = fetch_month(year, month)
        except AlpacaError as exc:
            print(f"{key}: ΣΦΑΛΜΑ — {exc}")
            print("Διακοπή. Ξανατρέξε το script για να συνεχίσει από εδώ.")
            break

        all_rows.extend(rows)
        completed.add(key)
        print(f"{key}: {len(rows):5d} άρθρα  (σύνολο: {len(all_rows)})")

        # Ανίχνευση αρχής ιστορικού: αν 12 συνεχόμενοι μήνες είναι άδειοι,
        # πιθανότατα δεν υπάρχουν δεδομένα τόσο πίσω.
        empty_streak = empty_streak + 1 if not rows else 0
        if empty_streak >= 12 and len(all_rows) == 0:
            print(f"\n12 συνεχόμενοι άδειοι μήνες — το ιστορικό δεν φτάνει στο {year}.")
            print("Άλλαξε το START_YEAR σε μεταγενέστερο έτος.")
            break

        # checkpoint κάθε μήνα
        pd.DataFrame(all_rows).to_csv(RAW_PATH, index=False)
        save_progress({"completed": sorted(completed), "empty_streak": empty_streak})
        time.sleep(PAUSE)

    df = pd.DataFrame(all_rows)
    if len(df):
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df = df.dropna(subset=["created_at"]).sort_values("created_at")
        df.to_csv(RAW_PATH, index=False)
        print(f"\nΣύνολο: {len(df)} άρθρα")
        print(f"Εύρος: {df['created_at'].min().date()} -> {df['created_at'].max().date()}")
        print(f"Μοναδικές ημέρες: {df['created_at'].dt.date.nunique()}")
    print(f"Αποθηκεύτηκε: {RAW_PATH}")

if __name__ == "__main__":
    download_all()