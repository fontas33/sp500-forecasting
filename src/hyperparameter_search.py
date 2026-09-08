"""
Hyperparameter search — επιλογή αποκλειστικά σε validation set.

Κρίσιμο σημείο μεθοδολογίας (διόρθωση A1 στην έρευνα): η κατάταξη των
συνδυασμών γίνεται ΜΟΝΟ με βάση balanced accuracy στο validation. Το test
set δεν αγγίζεται σε αυτό το στάδιο — μόνο στο evaluate.py, μία φορά.

Random search (Bergstra & Bengio, 2012) αντί εξαντλητικού grid.
"""

import itertools
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

from data_collection import load_snapshot
from features import build_features, FEATURES_TECH, FEATURES_MACRO, TARGET
from sentiment import attach_sentiment
from sequences import chronological_split, make_sequences, scale_splits
from train import train_model, predict, set_all_seeds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "validation_search_log.csv"
SELECTED_PATH = DATA_DIR / "selected_configs.csv"

GLOBAL_SEED = 42
SEEDS = [1, 2, 3]

FEATURE_SETS = {
    "tech7": FEATURES_TECH,
    "full10_alpaca": FEATURES_TECH + FEATURES_MACRO + ["SentimentMean"],
}

NN_SPACE = {"lookback": [30, 60, 90], "hidden": [32, 64, 128],
           "dropout": [0.1, 0.2, 0.3], "lr": [0.0005, 0.001, 0.005]}
N_NN_COMBOS = 10

FLAT_SPACE = {
    "LogReg": {"lookback": [30, 60], "C": [0.01, 0.1, 1.0]},
    "RF": {"lookback": [30, 60], "n_estimators": [300], "max_depth": [5, 10, None]},
    "XGB": {"lookback": [30, 60], "n_estimators": [300], "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1]},
}


def flatten(X):
    return X.reshape(X.shape[0], -1)


def build_flat_model(name, params, seed):
    if name == "LogReg":
        return LogisticRegression(C=params["C"], max_iter=2000, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=params["n_estimators"],
                                      max_depth=params["max_depth"],
                                      n_jobs=1, random_state=seed)
    return XGBClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                         learning_rate=params["learning_rate"], subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss",
                         random_state=seed, n_jobs=1, verbosity=0)


def bal_acc(y_true, y_pred_signal):
    return balanced_accuracy_score((y_true > 0).astype(int), (y_pred_signal > 0).astype(int))


def prepare_data():
    """
    Και τα δύο feature sets στο ΙΔΙΟ χρονικό εύρος (Alpaca coverage: 2015-2026),
    ώστε η σύγκριση να είναι δίκαιη.
    """
    from alpaca_sentiment import attach_alpaca_sentiment

    market = load_snapshot()
    base = build_features(market).dropna(subset=FEATURES_TECH + [TARGET]).reset_index(drop=True)

    with_sent = attach_alpaca_sentiment(base).dropna(
        subset=FEATURES_TECH + FEATURES_MACRO + ["SentimentMean"]
    ).reset_index(drop=True)

    # Το tech7 περιορίζεται στο ίδιο εύρος για δίκαιη σύγκριση
    common_range = base[base["Date"].isin(with_sent["Date"])].reset_index(drop=True)

    return {
        "tech7": chronological_split(common_range),
        "full10_alpaca": chronological_split(with_sent),
    }

def get_sequences(splits, feature_cols, lookback):
    train_df, val_df, test_df = splits
    X_tr, y_tr, d_tr = make_sequences(train_df, feature_cols, TARGET, lookback)
    X_va, y_va, d_va = make_sequences(val_df, feature_cols, TARGET, lookback)
    X_te, y_te, d_te = make_sequences(test_df, feature_cols, TARGET, lookback)
    X_tr_s, X_va_s, X_te_s, scaler = scale_splits(X_tr, X_va, X_te)
    return dict(Xtr=X_tr_s, ytr=y_tr, Xva=X_va_s, yva=y_va, Xte=X_te_s, yte=y_te,
               dtr=d_tr, dva=d_va, dte=d_te, scaler=scaler)


def run_search():
    print("Προετοιμασία δεδομένων...")
    all_splits = prepare_data()
    seq_cache = {}

    def cached_seqs(fs_name, lookback):
        key = (fs_name, lookback)
        if key not in seq_cache:
            seq_cache[key] = get_sequences(all_splits[fs_name], FEATURE_SETS[fs_name], lookback)
        return seq_cache[key]

    rng = random.Random(GLOBAL_SEED)
    all_combos = list(itertools.product(*NN_SPACE.values()))
    nn_combos = rng.sample(all_combos, N_NN_COMBOS)

    records = []
    t0 = time.time()

    # --- Νευρωνικά ---
    for fs_name in FEATURE_SETS:
        for arch in ["LSTM", "GRU"]:
            for ci, (lb, hidden, dropout, lr) in enumerate(nn_combos, 1):
                s = cached_seqs(fs_name, lb)
                accs = []
                for seed in SEEDS:
                    model = train_model(arch, s["Xtr"], s["ytr"], s["Xva"], s["yva"],
                                        hidden_size=hidden, dropout=dropout, lr=lr, seed=seed)
                    accs.append(bal_acc(s["yva"], predict(model, s["Xva"])))
                rec = {"feature_set": fs_name, "model": arch, "lookback": lb,
                      "hidden": hidden, "dropout": dropout, "lr": lr,
                      "val_bal_acc_mean": np.mean(accs), "val_bal_acc_std": np.std(accs)}
                records.append(rec)
                print(f"[{fs_name}/{arch}] {ci}/{len(nn_combos)} lb={lb} h={hidden} "
                      f"do={dropout} lr={lr} -> val={np.mean(accs)*100:.2f}%")

    # --- Κλασικά μοντέλα ---
    for fs_name in FEATURE_SETS:
        for mname, space in FLAT_SPACE.items():
            keys = [k for k in space if k != "lookback"]
            for lb in space["lookback"]:
                s = cached_seqs(fs_name, lb)
                Xtr_f, Xva_f = flatten(s["Xtr"]), flatten(s["Xva"])
                ytr_c, yva_c = (s["ytr"] > 0).astype(int), (s["yva"] > 0).astype(int)
                for combo in itertools.product(*[space[k] for k in keys]):
                    params = dict(zip(keys, combo))
                    accs = []
                    for seed in SEEDS:
                        set_all_seeds(seed)
                        clf = build_flat_model(mname, params, seed).fit(Xtr_f, ytr_c)
                        accs.append(balanced_accuracy_score(yva_c, clf.predict(Xva_f)))
                    rec = {"feature_set": fs_name, "model": mname, "lookback": lb,
                          "val_bal_acc_mean": np.mean(accs), "val_bal_acc_std": np.std(accs)}
                    rec.update({f"p_{k}": v for k, v in params.items()})
                    records.append(rec)
                    print(f"[{fs_name}/{mname}] lb={lb} {params} -> val={np.mean(accs)*100:.2f}%")

    print(f"\nΣυνολικός χρόνος: {(time.time()-t0)/60:.1f} λεπτά")

    search_df = pd.DataFrame(records)
    search_df.to_csv(LOG_PATH, index=False)

    winners = (search_df.sort_values("val_bal_acc_mean", ascending=False)
                        .groupby(["feature_set", "model"], as_index=False).first())
    winners.to_csv(SELECTED_PATH, index=False)

    print(f"\n=== Νικητές ανά (feature_set, model) — επιλογή σε VALIDATION ===")
    show = ["feature_set", "model", "lookback", "val_bal_acc_mean"]
    print(winners[[c for c in show if c in winners.columns]].to_string(index=False))
    print(f"\nΑποθηκεύτηκε: {LOG_PATH}\nΑποθηκεύτηκε: {SELECTED_PATH}")

    return winners


if __name__ == "__main__":
    run_search()