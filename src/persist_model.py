"""
Εκπαίδευση και αποθήκευση του production μοντέλου.

Αποθηκεύει ΜΑΖΙ: μοντέλο, scaler, και metadata. Χωρίς τον ίδιο scaler οι
live προβλέψεις είναι άκυρες — το μοντέλο θα έβλεπε είσοδο σε λάθος κλίμακα.

Υποστηρίζει και νευρωνικά (torch) και κλασικά (pickle) μοντέλα.
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from data_collection import load_snapshot
from features import build_features, FEATURES_TECH, FEATURES_MACRO, TARGET
from alpaca_sentiment import attach_alpaca_sentiment
from sequences import chronological_split, make_sequences, scale_splits
from hyperparameter_search import build_flat_model, flatten, FEATURE_SETS
from evaluate import coerce_params
from train import train_model, predict, set_all_seeds

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

PRODUCTION_SEED = 1
DEFAULT_FEATURE_SET = "full10_alpaca"


def build_dataset(feature_set):
    """Ίδια διαδικασία με το training — κρίσιμο για συνέπεια."""
    market = load_snapshot()
    base = build_features(market).dropna(subset=FEATURES_TECH + [TARGET]).reset_index(drop=True)

    if feature_set == "tech7":
        with_sent = attach_alpaca_sentiment(base)
        return base[base["Date"].isin(with_sent["Date"])].reset_index(drop=True)

    return attach_alpaca_sentiment(base).dropna(
        subset=FEATURES_TECH + FEATURES_MACRO + ["SentimentMean"]
    ).reset_index(drop=True)


def train_and_save(feature_set=DEFAULT_FEATURE_SET):
    winners = pd.read_csv(DATA_DIR / "selected_configs.csv")
    subset = winners[winners["feature_set"] == feature_set]
    if subset.empty:
        raise ValueError(f"Δεν βρέθηκε config για '{feature_set}'. "
                         f"Διαθέσιμα: {winners['feature_set'].unique().tolist()}")

    best = subset.sort_values("val_bal_acc_mean", ascending=False).iloc[0]
    model_name, lookback = best["model"], int(best["lookback"])
    features = FEATURE_SETS[feature_set]
    out_dir = MODEL_DIR / feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Production model: {feature_set}/{model_name}, lookback={lookback}, "
          f"val_bal_acc={best['val_bal_acc_mean']*100:.2f}%")

    data = build_dataset(feature_set)
    train_df, val_df, test_df = chronological_split(data)

    X_tr, y_tr, _ = make_sequences(train_df, features, TARGET, lookback)
    X_va, y_va, _ = make_sequences(val_df, features, TARGET, lookback)
    X_te, y_te, _ = make_sequences(test_df, features, TARGET, lookback)
    X_tr_s, X_va_s, X_te_s, scaler = scale_splits(X_tr, X_va, X_te)

    is_neural = model_name in ("LSTM", "GRU")

    if is_neural:
        model = train_model(model_name, X_tr_s, y_tr, X_va_s, y_va,
                            hidden_size=int(best["hidden"]), dropout=float(best["dropout"]),
                            lr=float(best["lr"]), seed=PRODUCTION_SEED)
        torch.save(model.state_dict(), out_dir / "production_model.pt")
        test_pred = predict(model, X_te_s)
        params = {"hidden_size": int(best["hidden"]), "dropout": float(best["dropout"]),
                  "lr": float(best["lr"]), "n_features": X_tr.shape[2]}
    else:
        raw = {k[2:]: best[k] for k in best.index
               if str(k).startswith("p_") and pd.notna(best[k])}
        params = coerce_params(raw)
        if model_name == "RF" and "max_depth" not in params:
            params["max_depth"] = None
        set_all_seeds(PRODUCTION_SEED)
        model = build_flat_model(model_name, params, PRODUCTION_SEED)
        model.fit(flatten(X_tr_s), (y_tr > 0).astype(int))
        with open(out_dir / "production_model.pkl", "wb") as f:
            pickle.dump(model, f)
        test_pred = model.predict_proba(flatten(X_te_s))[:, 1] - 0.5

    with open(out_dir / "production_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    acc = balanced_accuracy_score((y_te > 0).astype(int), (test_pred > 0).astype(int))

    metadata = {
        "model_type": model_name,
        "is_neural": is_neural,
        "feature_set": feature_set,
        "features": features,
        "lookback": lookback,
        "params": {k: (None if v is None else v) for k, v in params.items()},
        "seed": PRODUCTION_SEED,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_range": [str(train_df["Date"].min().date()), str(train_df["Date"].max().date())],
        "test_bal_acc": float(acc),
        "signal_threshold": 0.0,
        "max_sentiment_staleness_days": 5,
        "notes": [
            "Το sentiment προέρχεται από το Alpaca news API (live-συμβατή πηγή).",
            "Κενές ημέρες sentiment: forward-fill, ίδια λογική με το training.",
            "Καμία στατιστικά σημαντική προγνωστική ικανότητα — βλ. αξιολόγηση.",
        ],
    }
    with open(out_dir / "production_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Sanity check — test balanced accuracy: {acc*100:.2f}%")
    print(f"\nΑποθηκεύτηκαν στο {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default=DEFAULT_FEATURE_SET,
                    choices=list(FEATURE_SETS))
    args = ap.parse_args()
    train_and_save(args.feature_set)