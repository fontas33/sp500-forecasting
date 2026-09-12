"""
Walk-forward evaluation στο tech7 feature set (πλήρες ιστορικό 1993-2026).

Πέντε expanding-window folds: κάθε fold εκπαιδεύεται σε όλα τα δεδομένα μέχρι
μια χρονική στιγμή και αξιολογείται στην επόμενη περίοδο. Έτσι κάθε fold
σέβεται τη χρονική αιτιότητα, ενώ συνολικά καλύπτονται διαφορετικά καθεστώτα
αγοράς (κρίση 2008, COVID 2020, ανοδικές περίοδοι).

Το full10_alpaca ΔΕΝ συμπεριλαμβάνεται: η πηγή ειδήσεων ξεκινά το 2015, οπότε
δεν υπάρχει αρκετό ιστορικό για ουσιαστικά folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from data_collection import load_snapshot
from features import build_features, FEATURES_TECH, TARGET
from sequences import make_sequences
from train import train_model, predict, set_all_seeds
from backtest import perf_metrics, run_signal, permutation_test_sharpe, TRADING_DAYS, COST

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_PATH = DATA_DIR / "walkforward_results.csv"

SEEDS = [1, 2, 3]
ARCHS = ["LSTM", "GRU"]
THRESHOLD_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003]

# Κάθε fold: (τέλος train, τέλος validation, τέλος test)
# Το validation χρησιμεύει για early stopping ΚΑΙ επιλογή threshold.
FOLDS = [
    ("2005-12-31", "2007-06-30", "2009-12-31"),   # κρίση 2008
    ("2009-12-31", "2011-06-30", "2013-12-31"),   # ανάκαμψη
    ("2013-12-31", "2015-06-30", "2017-12-31"),   # ήρεμη ανοδική
    ("2017-12-31", "2019-06-30", "2021-12-31"),   # COVID crash & rebound
    ("2021-12-31", "2023-06-30", "2026-12-31"),   # πρόσφατη
]

# Υπερπαράμετροι: παγωμένες από το κύριο pipeline, ΔΕΝ ξανα-συντονίζονται ανά
# fold (θα ήταν 5x περισσότερες συγκρίσεις και ανεξέλεγκτο data snooping).
FIXED_PARAMS = {
    "LSTM": {"lookback": 60, "hidden": 64, "dropout": 0.2, "lr": 0.001},
    "GRU":  {"lookback": 60, "hidden": 64, "dropout": 0.2, "lr": 0.001},
}


def split_by_date(data, train_end, val_end, test_end):
    train = data[data["Date"] <= train_end].reset_index(drop=True)
    val = data[(data["Date"] > train_end) & (data["Date"] <= val_end)].reset_index(drop=True)
    test = data[(data["Date"] > val_end) & (data["Date"] <= test_end)].reset_index(drop=True)
    return train, val, test


def prepare_fold(data, fold, lookback):
    train_df, val_df, test_df = split_by_date(data, *fold)
    if min(len(train_df), len(val_df), len(test_df)) <= lookback + 20:
        return None

    X_tr, y_tr, d_tr = make_sequences(train_df, FEATURES_TECH, TARGET, lookback)
    X_va, y_va, d_va = make_sequences(val_df, FEATURES_TECH, TARGET, lookback)
    X_te, y_te, d_te = make_sequences(test_df, FEATURES_TECH, TARGET, lookback)

    n_feat = X_tr.shape[2]
    scaler = StandardScaler().fit(X_tr.reshape(-1, n_feat))
    tf = lambda A: scaler.transform(A.reshape(-1, n_feat)).reshape(A.shape)

    return dict(Xtr=tf(X_tr), ytr=y_tr, Xva=tf(X_va), yva=y_va, Xte=tf(X_te), yte=y_te,
                dva=d_va, dte=d_te,
                train_range=(train_df["Date"].min(), train_df["Date"].max()),
                test_range=(test_df["Date"].min(), test_df["Date"].max()))


def run():
    market = load_snapshot()
    data = (build_features(market)
            .dropna(subset=FEATURES_TECH + [TARGET, "RiskFreeDaily"])
            .reset_index(drop=True))
    rf_map = data.set_index("Date")["RiskFreeDaily"]
    print(f"Δεδομένα: {len(data)} γραμμές, "
          f"{data['Date'].min().date()} -> {data['Date'].max().date()}\n")

    rows = []
    for fi, fold in enumerate(FOLDS, 1):
        for arch in ARCHS:
            p = FIXED_PARAMS[arch]
            s = prepare_fold(data, fold, p["lookback"])
            if s is None:
                print(f"Fold {fi} / {arch}: παραλείπεται (ανεπαρκή δεδομένα)")
                continue

            # --- ταξινόμηση: 3 seeds ---
            accs, val_preds, test_preds = [], [], []
            for seed in SEEDS:
                model = train_model(arch, s["Xtr"], s["ytr"], s["Xva"], s["yva"],
                                    hidden_size=p["hidden"], dropout=p["dropout"],
                                    lr=p["lr"], seed=seed)
                vp, tp = predict(model, s["Xva"]), predict(model, s["Xte"])
                val_preds.append(vp)
                test_preds.append(tp)
                accs.append(balanced_accuracy_score((s["yte"] > 0).astype(int),
                                                    (tp > 0).astype(int)))

            # --- backtest με seed 1, threshold επιλεγμένο σε validation ---
            rf_val = rf_map.reindex(pd.to_datetime(s["dva"])).ffill().fillna(0).values
            rf_test = rf_map.reindex(pd.to_datetime(s["dte"])).ffill().fillna(0).values

            best_t, best_sharpe = 0.0, -np.inf
            for t in THRESHOLD_GRID:
                sr, sig, _ = run_signal(val_preds[0], s["yva"], rf_val, t)
                m = perf_metrics(sr, rf_val, "val", sig.mean())
                if np.isfinite(m["Sharpe"]) and m["Sharpe"] > best_sharpe:
                    best_sharpe, best_t = m["Sharpe"], t

            strat_r, sig, n_trades = run_signal(test_preds[0], s["yte"], rf_test, best_t)
            strat_m = perf_metrics(strat_r, rf_test, "strategy", sig.mean(), n_trades)
            bh_m = perf_metrics(s["yte"], rf_test, "buyhold", 1.0, 1)

            p_val, null_dist = permutation_test_sharpe(
                s["yte"], rf_test, strat_m["exposure"], strat_m["Sharpe"]
            )

            sigma_ratio = test_preds[0].std() / s["yte"].std()

            rows.append({
                "fold": fi, "arch": arch,
                "train_start": s["train_range"][0].date(), "train_end": s["train_range"][1].date(),
                "test_start": s["test_range"][0].date(), "test_end": s["test_range"][1].date(),
                "n_test": len(s["yte"]),
                "bal_acc_mean": np.mean(accs), "bal_acc_std": np.std(accs),
                "threshold": best_t,
                "strat_sharpe": strat_m["Sharpe"], "bh_sharpe": bh_m["Sharpe"],
                "strat_return": strat_m["total_return"], "bh_return": bh_m["total_return"],
                "strat_mdd": strat_m["max_drawdown"], "bh_mdd": bh_m["max_drawdown"],
                "exposure": strat_m["exposure"], "n_trades": n_trades,
                "perm_p": p_val, "sigma_ratio": sigma_ratio,
            })

            print(f"Fold {fi} [{s['test_range'][0].date()}->{s['test_range'][1].date()}] "
                  f"{arch}: bal_acc={np.mean(accs)*100:.2f}% | "
                  f"Sharpe {strat_m['Sharpe']:+.3f} vs B&H {bh_m['Sharpe']:+.3f} | "
                  f"perm p={p_val:.3f} | σ_ratio={sigma_ratio:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)

    # --- σύνοψη ---
    print("\n" + "=" * 78)
    print("ΣΥΝΟΨΗ WALK-FORWARD")
    print("=" * 78)
    for arch in ARCHS:
        sub = df[df["arch"] == arch]
        if not len(sub):
            continue
        print(f"\n{arch} ({len(sub)} folds):")
        print(f"  Balanced accuracy : {sub['bal_acc_mean'].mean()*100:.2f}% "
              f"± {sub['bal_acc_mean'].std()*100:.2f}  "
              f"(εύρος {sub['bal_acc_mean'].min()*100:.1f}-{sub['bal_acc_mean'].max()*100:.1f}%)")
        print(f"  Folds > 50%       : {(sub['bal_acc_mean'] > 0.5).sum()}/{len(sub)}")
        print(f"  Strategy Sharpe   : {sub['strat_sharpe'].mean():+.3f} "
              f"(B&H: {sub['bh_sharpe'].mean():+.3f})")
        print(f"  Folds Sharpe>B&H  : {(sub['strat_sharpe'] > sub['bh_sharpe']).sum()}/{len(sub)}")

    # Bonferroni στα permutation tests
    n_tests = len(df)
    alpha_bonf = 0.05 / n_tests if n_tests else np.nan
    sig = df[df["perm_p"] < alpha_bonf]
    print(f"\n--- Permutation tests: {n_tests} συγκρίσεις, "
          f"Bonferroni α={alpha_bonf:.4f} ---")
    print(f"Folds με p < 0.05 (αδιόρθωτο) : {(df['perm_p'] < 0.05).sum()}/{n_tests}")
    print(f"Folds που επιβιώνουν Bonferroni: {len(sig)}/{n_tests}")

    print(f"\nΑποθηκεύτηκε: {RESULTS_PATH}")
    return df


if __name__ == "__main__":
    run()