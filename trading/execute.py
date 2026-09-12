"""
Εκτέλεση εντολών στο Alpaca paper trading.

ΑΣΦΑΛΕΙΑ:
- Μόνο paper endpoint (ελέγχεται ρητά στο alpaca_client)
- Dry-run από προεπιλογή· χρειάζεται ρητό --live για αποστολή
- Μόνο long/cash — καμία short θέση, καμία μόχλευση
- Ένα σύμβολο (SPY), σταθερό ποσοστό κεφαλαίου
- Κάθε ενέργεια καταγράφεται πριν και μετά
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "trading"))

from alpaca_client import (BASE_URL, HEADERS, AlpacaError, _check_credentials,
                           get_account, get_positions, get_clock)
from signal_generator import generate_signal, log_signal

SYMBOL = "SPY"
TARGET_ALLOCATION = 0.95          # % του κεφαλαίου όταν LONG (όχι 100%, περιθώριο)
TRADE_LOG = ROOT / "data" / "trade_log.csv"


def _post(endpoint, payload):
    _check_credentials()
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=15)
    except requests.exceptions.RequestException as exc:
        raise AlpacaError(f"Αποτυχία σύνδεσης: {type(exc).__name__}: {exc}") from exc

    if r.status_code == 403:
        raise AlpacaError(f"403: Απορρίφθηκε — {r.text[:200]}")
    if r.status_code == 422:
        raise AlpacaError(f"422: Μη έγκυρη εντολή — {r.text[:200]}")
    if not r.ok:
        raise AlpacaError(f"{r.status_code}: {r.text[:200]}")
    return r.json()


def submit_order(side, qty):
    return _post("orders", {
        "symbol": SYMBOL, "qty": str(qty), "side": side,
        "type": "market", "time_in_force": "day",
    })


def current_position_qty():
    for p in get_positions():
        if p["symbol"] == SYMBOL:
            return float(p["qty"])
    return 0.0


def decide_action(signal, current_qty, equity, last_close):
    """
    Μεταφράζει το σήμα σε συγκεκριμένη ενέργεια.

    Επιστρέφει (side, qty, reason) ή (None, 0, reason) αν δεν χρειάζεται τίποτα.
    """
    target_qty = int((equity * TARGET_ALLOCATION) // last_close) if signal else 0
    delta = target_qty - int(current_qty)

    if delta == 0:
        return None, 0, f"Ήδη στη σωστή θέση ({int(current_qty)} μετοχές)"
    if delta > 0:
        return "buy", delta, f"Αύξηση θέσης {int(current_qty)} -> {target_qty}"
    return "sell", abs(delta), f"Μείωση θέσης {int(current_qty)} -> {target_qty}"


def log_trade(record):
    row = pd.DataFrame([record])
    if TRADE_LOG.exists():
        row.to_csv(TRADE_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(TRADE_LOG, index=False)


def run(live=False, force=False):
    clock = get_clock()
    if not clock["is_open"] and not force:
        print(f"Η αγορά είναι κλειστή. Επόμενο άνοιγμα: {clock['next_open']}")
        print("Χρησιμοποίησε --force για δοκιμή εκτός ωραρίου (η εντολή θα μείνει "
              "σε queue μέχρι το άνοιγμα).")
        return

    sig = generate_signal()
    log_signal(sig)

    account = get_account()
    if account["trading_blocked"]:
        raise AlpacaError("Ο λογαριασμός έχει μπλοκαρισμένο trading")

    equity = float(account["equity"])
    current_qty = current_position_qty()
    side, qty, reason = decide_action(sig["signal"], current_qty,
                                      equity, sig["last_close"])

    print(f"\n=== ΑΠΟΦΑΣΗ ===")
    print(f"  Σήμα            : {sig['action']} (raw={sig['raw_prediction']:+.6f})")
    print(f"  Κεφάλαιο        : ${equity:,.2f}")
    print(f"  Τρέχουσα θέση   : {int(current_qty)} {SYMBOL}")
    print(f"  Ενέργεια        : {reason}")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "signal": sig["signal"], "raw_prediction": sig["raw_prediction"],
        "equity": equity, "current_qty": current_qty,
        "side": side or "none", "qty": qty, "reason": reason,
        "mode": "live" if live else "dry-run", "order_id": None, "status": None,
    }

    if side is None:
        print("\nΚαμία εντολή απαραίτητη.")
        log_trade(record)
        return

    if not live:
        print(f"\n[DRY-RUN] Θα στελνόταν: {side.upper()} {qty} {SYMBOL}")
        print("Για πραγματική αποστολή (paper account): --live")
        log_trade(record)
        return

    print(f"\nΑποστολή εντολής: {side.upper()} {qty} {SYMBOL}...")
    order = submit_order(side, qty)
    record["order_id"] = order.get("id")
    record["status"] = order.get("status")
    print(f"  Order ID : {order.get('id')}")
    print(f"  Status   : {order.get('status')}")
    log_trade(record)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Στέλνει πραγματική εντολή (paper account)")
    ap.add_argument("--force", action="store_true",
                    help="Εκτέλεση και με κλειστή αγορά")
    args = ap.parse_args()

    try:
        run(live=args.live, force=args.force)
    except (AlpacaError, Exception) as exc:
        print(f"\nΣΦΑΛΜΑ: {type(exc).__name__}: {exc}")
        sys.exit(1)