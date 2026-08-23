"""
Τελική αποτίμηση στο test set — μία φορά, χωρίς επιλογή εδώ.

Οι διαμορφώσεις είναι ήδη παγωμένες (από το hyperparameter_search.py,
επιλεγμένες αποκλειστικά σε validation). Αυτό το script μόνο αποτιμά.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

from hyperparameter_search import (
    FEATURE_SETS, build_flat_model, flatten, bal_acc, prepare_data, get_sequences,
)
from train import train_model, predict, set_all_seeds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SELECTED_PATH = DATA_DIR / "selected_configs.csv"
RESULTS_PATH = DATA_DIR / "final_test_results.csv"

FINAL_SEEDS = [1, 2, 3, 4, 5]
INT_PARAMS = {"n_estimators", "max_depth"}


def coerce_params(d):
    """Επαναφορά σωστών τύπων: pandas μετατρέπει ints σε floats όταν η στήλη έχει NaN."""
    out = {}
    for k, v in d.items():
        if k in INT_PARAMS:
            out[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else int(v)
        else:
            out[k] = float(v)
    return out


def evaluate_all():
    winners = pd.read_csv(SELECTED_PATH)
    all_splits = prepare_data()
    seq_cache = {}

    def cached_seqs(fs_name, lookback):
        key = (fs_name, lookback)
        if key not in seq_cache:
            seq_cache[key] = get_sequences(all_splits[fs_name], FEATURE_SETS[fs_name], lookback)
        return seq_cache[key]

    rows, stored_preds = [], {}

    for _, w in winners.iterrows():
        fs_name, mname, lb = w["feature_set"], w["model"], int(w["lookback"])
        s = cached_seqs(fs_name, lb)
        preds_per_seed = []

        if mname in ("LSTM", "GRU"):
            for seed in FINAL_SEEDS:
                model = train_model(mname, s["Xtr"], s["ytr"], s["Xva"], s["yva"],
                                    hidden_size=int(w["hidden"]), dropout=float(w["dropout"]),
                                    lr=float(w["lr"]), seed=seed)
                preds_per_seed.append(predict(model, s["Xte"]))
        else:
            Xtr_f, Xte_f = flatten(s["Xtr"]), flatten(s["Xte"])
            ytr_c = (s["ytr"] > 0).astype(int)
            raw = {k[2:]: w[k] for k in w.index if str(k).startswith("p_") and pd.notna(w[k])}
            params = coerce_params(raw)
            if mname == "RF" and "max_depth" not in params:
                params["max_depth"] = None
            for seed in FINAL_SEEDS:
                set_all_seeds(seed)
                clf = build_flat_model(mname, params, seed).fit(Xtr_f, ytr_c)
                preds_per_seed.append(clf.predict_proba(Xte_f)[:, 1] - 0.5)

        P = np.vstack(preds_per_seed)
        accs = [bal_acc(s["yte"], p) for p in P]
        pups = [(p > 0).mean() for p in P]

        stored_preds[(fs_name, mname)] = dict(P=P, y=s["yte"], dates=s["dte"], lookback=lb)
        rows.append({"feature_set": fs_name, "model": mname, "lookback": lb,
                     "val_bal_acc": w["val_bal_acc_mean"],
                     "test_bal_acc_mean": np.mean(accs), "test_bal_acc_std": np.std(accs),
                     "pct_up_mean": np.mean(pups)})
        print(f"[{fs_name}/{mname}] val={w['val_bal_acc_mean']*100:.2f}% -> "
              f"TEST={np.mean(accs)*100:.2f}% ±{np.std(accs)*100:.2f}  %Up={np.mean(pups)*100:.1f}%")

    results_df = pd.DataFrame(rows).sort_values("test_bal_acc_mean", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)
    return results_df, stored_preds


def bootstrap_ci(y_true, pred, n_boot=2000, alpha=0.05, seed=42):
    rs = np.random.RandomState(seed)
    yt, yp = (y_true > 0).astype(int), (pred > 0).astype(int)
    n = len(yt)
    stats_ = [balanced_accuracy_score(yt[idx], yp[idx])
             for idx in (rs.randint(0, n, n) for _ in range(n_boot))]
    return np.percentile(stats_, [alpha / 2 * 100, (1 - alpha / 2) * 100])


def mcnemar_test(y_true, pred_a, pred_b):
    yt = (y_true > 0).astype(int)
    a = (pred_a > 0).astype(int) == yt
    b = (pred_b > 0).astype(int) == yt
    tbl = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
    return mcnemar(tbl, exact=False, correction=True).pvalue


def diagnostics(stored_preds):
    print("\n=== Bootstrap 95% CI (seed 1) ===")
    for (fs, mn), d in stored_preds.items():
        p = d["P"][0]
        lo, hi = bootstrap_ci(d["y"], p)
        point = bal_acc(d["y"], p)
        flag = "  <-- περιλαμβάνει 50%" if lo <= 0.5 <= hi else ""
        print(f"{fs:7s} {mn:7s}: {point*100:5.2f}%  CI[{lo*100:5.2f}, {hi*100:5.2f}]{flag}")

    n_comparisons = len(stored_preds)
    bonf_alpha = 0.05 / n_comparisons
    print(f"\n=== Bonferroni-corrected έλεγχος (n={n_comparisons} συγκρίσεις, "
          f"διορθωμένο α={bonf_alpha:.4f}) ===")
    for (fs, mn), d in stored_preds.items():
        p = d["P"][0]
        lo, hi = bootstrap_ci(d["y"], p, alpha=bonf_alpha)
        crosses = lo <= 0.5 <= hi
        verdict = "μη σημαντική" if crosses else "ΣΗΜΑΝΤΙΚΗ (επιβιώνει διόρθωση)"
        print(f"{fs:7s} {mn:7s}: {bonf_alpha*100/2:.2f}%-{100-bonf_alpha*100/2:.2f}% "
              f"CI[{lo*100:5.2f}, {hi*100:5.2f}]  -> {verdict}")
    
    print("\n=== McNemar: tech7 vs full10 ανά αρχιτεκτονική (seed 1) ===")
    for mn in ["LSTM", "GRU", "LogReg", "RF", "XGB"]:
        ka, kb = ("tech7", mn), ("full10", mn)
        if ka in stored_preds and kb in stored_preds:
            da, db = stored_preds[ka], stored_preds[kb]
            L = min(len(da["y"]), len(db["y"]))
            p = mcnemar_test(da["y"][-L:], da["P"][0][-L:], db["P"][0][-L:])
            verdict = "ΣΗΜΑΝΤΙΚΗ" if p < 0.05 else "μη σημαντική"
            print(f"{mn:7s}: p = {p:.4f}  -> {verdict}")

    print("\n=== Διάγνωση mode collapse (μόνο LSTM/GRU — ίδιες μονάδες με actual) ===")
    for (fs, mn), d in stored_preds.items():
        if mn not in ("LSTM", "GRU"):
            continue
        p, y = d["P"][0], d["y"]
        ratio = p.std() / y.std()
        flag = "  <-- COLLAPSE" if ratio < 0.20 else ""
        print(f"{fs}/{mn}: σ(pred)/σ(actual) = {ratio:.3f}{flag}")

    print("\n=== Διάγνωση μεροληψίας κλάσης (LogReg/RF/XGB - pct_up ως ένδειξη) ===")
    for (fs, mn), d in stored_preds.items():
        if mn in ("LSTM", "GRU"):
            continue
        pct_up = (d["P"][0] > 0).mean()
        skew = abs(pct_up - 0.5)
        flag = " <-- ΙΣΧΥΡΗ ΜΕΡΟΛΗΨΙΑ" if skew > 0.25 else ""
        print(f"{fs}/{mn}: pct_up = {pct_up*100:.1f}%{flag}")

if __name__ == "__main__":
    results_df, stored_preds = evaluate_all()
    print(f"\n{results_df.to_string(index=False)}")
    diagnostics(stored_preds)
    print(f"\nΑποθηκεύτηκε: {RESULTS_PATH}")