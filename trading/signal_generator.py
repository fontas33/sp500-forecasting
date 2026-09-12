"""
Παραγωγή σήματος trading από το production μοντέλο.

Φορτώνει το παγωμένο μοντέλο και τον ΑΠΟΘΗΚΕΥΜΕΝΟ scaler του training.
Δεν στέλνει εντολές — μόνο υπολογίζει τι θα έπρεπε να γίνει.
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "trading"))

from models import ARCHITECTURES
from live_features import build_live_window, load_metadata, LiveFeatureError

MODEL_DIR = ROOT / "models"
LOG_PATH = ROOT / "data" / "signal_log.csv"


def load_production_model(meta):
    """Φορτώνει μοντέλο + scaler από τα αποθηκευμένα artifacts."""
    with open(MODEL_DIR / "production_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    if meta["is_neural"]:
        params = meta["params"]
        Model = ARCHITECTURES[meta["model_type"]]
        model = Model(n_features=params["n_features"],
                      hidden_size=params["hidden_size"],
                      dropout=params["dropout"])
        model.load_state_dict(torch.load(MODEL_DIR / "production_model.pt",
                                         map_location="cpu"))
        model.eval()
    else:
        with open(MODEL_DIR / "production_model.pkl", "rb") as f:
            model = pickle.load(f)

    return model, scaler


def generate_signal():
    """
    Επιστρέφει dict με το σήμα και όλο το context.

    Το σήμα είναι 1 (long) ή 0 (cash), βάσει του threshold των metadata.
    """
    meta = load_metadata()
    X, info = build_live_window()
    model, scaler = load_production_model(meta)

    n_feat = X.shape[2]
    X_scaled = scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape)

    if meta["is_neural"]:
        with torch.no_grad():
            raw = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().ravel()[0]
    else:
        flat = X_scaled.reshape(X_scaled.shape[0], -1)
        raw = float(model.predict_proba(flat)[:, 1] - 0.5)

    threshold = meta.get("signal_threshold", 0.0)
    signal = int(raw > threshold)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": info["as_of_date"],
        "last_close": info["last_close"],
        "model": f"{meta['feature_set']}/{meta['model_type']}",
        "raw_prediction": float(raw),
        "threshold": threshold,
        "signal": signal,
        "action": "LONG" if signal else "CASH",
        **{k: v for k, v in info.items() if k.startswith("staleness")
           or k.startswith("last_real")},
    }


def log_signal(sig):
    """Καταγραφή κάθε σήματος — απαραίτητο για audit trail."""
    import pandas as pd
    row = pd.DataFrame([sig])
    if LOG_PATH.exists():
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    try:
        sig = generate_signal()
    except LiveFeatureError as exc:
        print(f"ΔΕΝ ΠΑΡΑΓΕΤΑΙ ΣΗΜΑ: {exc}")
        sys.exit(1)

    print("=== ΣΗΜΑ TRADING ===\n")
    for k, v in sig.items():
        print(f"  {k:18s}: {v}")

    log_signal(sig)
    print(f"\nΚαταγράφηκε στο {LOG_PATH.name}")

    print("\nΥπενθύμιση: το μοντέλο ΔΕΝ έχει αποδεδειγμένη προγνωστική ικανότητα "
          "(βλ. αξιολόγηση). Το σήμα παράγεται για επίδειξη λειτουργίας.")