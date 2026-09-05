"""
Σύνδεση με το Alpaca Trading API (paper trading).
Τα credentials διαβάζονται από το .env, το οποίο ΔΕΝ ανεβαίνει ποτέ στο Git.
Όλες οι κλήσεις πηγαίνουν στο paper endpoint - καμία πραγματική συναλλαγή.
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,   
}

class AlpacaError(Exception):
    """Σφάλμα κατά την επικοινωνία με το Alpaca API."""

def _check_credentials():
    missing = [name for name, val in
               [("ALPACA_API_KEY", API_KEY),
                ("ALPACA_SECRET_KEY", SECRET_KEY),
                ("ALPACA_BASE_URL", BASE_URL)] if not val]
    if missing:
        raise AlpacaError(
            f"Λείπουν από το .env: {', '.join(missing)}. "
            f"Έλεγξε ότι υπάρχει το αρχείο {PROJECT_ROOT / '.env'}"
        )
    if "paper-api" not in BASE_URL:
        raise AlpacaError(
            f"ΠΡΟΣΟΧΗ: το BASE_URL δεν δείχνει σε paper endpoint ({BASE_URL})."
            "Σταματάω για ασφάλεια."
        )

def _get(endpoint):
    """
    Εκτελεί GET request και επιστρέφει το JSON.
    
    Ο έλεγχος status code είναι ρητός: σε trading σύστημα, η διαφορά ανάμεσα σε "απέτυχε" και "δεν ξέρω αν πέτυχε" έχει σημασία.
    """
    _check_credentials()
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
    except requests.exceptions.RequestException as exc:
        raise AlpacaError(f"Αποτυχία σύνδεσης: {type(exc).__name__}: {exc}") from exc

    if response.status_code == 401:
        raise AlpacaError("401: Μη έγκυρα credentials - έλεγξε τα κλειδιά στο .env")
    if response.status_code == 403:
        raise AlpacaError("403: Η ενέργεια δεν επιτρέπεταιγια αυτόν τον λογαριασμό")
    if response.status_code == 429:
        raise AlpacaError("429: Υπέρβαση ορίου αιτημάτων (rate limit) - περίμενε λίγο")
    if response.status_code >= 500:
        raise AlpacaError(f"{response.status_code}: Σφάλμα στον server του Alpaca")
    if not response.ok:
        raise AlpacaError(f"{response.status_code}: {response.text[:200]}")

    return response.json()

def get_account():
    """Στοιχεία λογαριασμού: κεφάλαιο, buying power, κατάσταση."""
    return _get("account")

def get_positions():
    """Τρέχουσες ανοιχτές θέσεις."""
    return _get("positions")

def get_clock():
    """Κατάσταση αγοράς: ανοιχτή/κλειστή, επόμενο άνοιγμα/κλείσιμο."""
    return _get("clock")

if __name__ == "__main__":
    print("Έλεγχος σύνδεσης με Alpaca paper trading...\n")

    account = get_account()
    print(f"Λογαριασμός:      {account['account_number']}")
    print(f"Κατάσταση:        {account['status']}")
    print(f"Κεφάλαιο:         ${float(account['equity']):,.2f}")
    print(f"Buying power:     ${float(account['buying_power']):,.2f}")
    print(f"Trading blocked:  {account['trading_blocked']}")

    clock = get_clock()
    print(f"\nΑγορά ανοιχτή:    {clock['is_open']}")
    print(f"Επόμενο άνοιγμα:  {clock['next_open']}")

    positions = get_positions()
    print(f"\nΑνοιχτές θέσεις:  {len(positions)}")
    for p in positions:
        print(f"  {p['symbol']}: {p['qty']} @ ${float(p['avg_entry_price']):,.2f}")

def get_news(symbols="SPY", start=None, end=None, limit=50, page_token=None):
    """
    Ειδήσεις από το Alpaca news API.
    
    Σημείωση: το news endpoint είναι σε διαφορετικό host από το trading API.
    """
    _check_credentials()
    url = "https://data.alpaca.markets/v1beta1/news"
    params = {"symbols": symbols, "limit": limit}
    if page_token:
        params["page_token"] = page_token
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    response = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if not response.ok:
        raise AlpacaError(f"{response.status_code}: {response.text[:300]}")
    return response.json()