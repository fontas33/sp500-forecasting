"""
Backtesting με risk-adjusted metrics, σε πολλαπλά μοντέλα.

Παραδοχές εκτέλεσης (δηλωμένες ρητά):
1. Εκτέλεση ακριβώς στο κλείσιμο· close-to-close αποδόσεις.
2. Κόστος 0.1% ανά αλλαγή θέσης· χωρίς φόρους, market impact.
3. Cash αποδίδει το πραγματικό risk-free (3M T-bill), όχι 0%.
4. Threshold επιλέγεται ΑΠΟΚΛΕΙΣΤΙΚΑ στο validation (μέγιστο Sharpe),
  ποτέ στο test.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_collection import load_snapshot
from features import build_features, FEATURES_TECH, TARGET
from hyperparameter_search import (
    FEATURE_SETS, build_flat_model, flatten, prepare_data, get_sequences,
)
from evaluate import coerce_params
from train import train_model, predict, set_all_seeds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SELECTED_PATH = DATA_DIR / "selected_configs.csv"
RESULTS_PATH = DATA_DIR / "backtest_results.csv"

TRADING_DAYS = 252
COST = 0.001
N_MODELS_TO_TEST = 3
THRESHOLD_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003]


def perf_metrics(returns, rf_series, label, exposure=None, n_trades=None):
    r = np.asarray(returns, dtype=float)
    rf = np.asarray(rf_series, dtype=float)
    excess = r - rf
    cum = np.cumprod(1 + r)
    years = len(r) / TRADING_DAYS
    cagr = cum[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
             if excess.std(ddof=1) > 0 else np.nan)
    downside = excess[excess < 0]
    sortino = (excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS)
              if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min()
    return {"strategy": label, "total_return": cum[-1] - 1, "CAGR": cagr,
           "ann_vol": vol, "Sharpe": sharpe, "Sortino": sortino,
           "max_drawdown": mdd, "exposure": exposure, "n_trades": n_trades}


def run_signal(pred, y_actual, rf, threshold):
    signal = (pred > threshold).astype(float)
    changes = np.abs(np.diff(np.concatenate([[0.0], signal])))
    strat = signal * y_actual + (1 - signal) * rf - changes * COST
    return strat, signal, int(changes.sum())


def get_predictions(row, splits_cache, feature_sets):
    """Ξαναπαράγει προβλέψεις (seed=1) για ένα validation-selected config."""
    fs_name, mname, lb = row["feature_set"], row["model"], int(row["lookback"])
    s = splits_cache[(fs_name, lb)]

    if mname in ("LSTM", "GRU"):
        model = train_model(mname, s["Xtr"], s["ytr"], s["Xva"], s["yva"],
                            hidden_size=int(row["hidden"]), dropout=float(row["dropout"]),
                            lr=float(row["lr"]), seed=1)
        val_pred, test_pred = predict(model, s["Xva"]), predict(model, s["Xte"])
    else:
        raw = {k[2:]: row[k] for k in row.index if str(k).startswith("p_") and pd.notna(row[k])}
        params = coerce_params(raw)
        if mname == "RF" and "max_depth" not in params:
            params["max_depth"] = None
        set_all_seeds(1)
        clf = build_flat_model(mname, params, 1).fit(flatten(s["Xtr"]), (s["ytr"] > 0).astype(int))
        val_pred = clf.predict_proba(flatten(s["Xva"]))[:, 1] - 0.5
        test_pred = clf.predict_proba(flatten(s["Xte"]))[:, 1] - 0.5

    return val_pred, test_pred, s


def backtest_one(row, splits_cache, rf_map):
    val_pred, test_pred, s = get_predictions(row, splits_cache, FEATURE_SETS)

    rf_val = rf_map.reindex(pd.to_datetime(s["dva"])).ffill().fillna(0).values
    rf_test = rf_map.reindex(pd.to_datetime(s["dte"])).ffill().fillna(0).values

    # --- threshold: επιλογή ΑΠΟΚΛΕΙΣΤΙΚΑ σε validation ---
    best_t, best_sharpe = 0.0, -np.inf
    for t in THRESHOLD_GRID:
        sr, sig, _ = run_signal(val_pred, s["yva"], rf_val, t)
        m = perf_metrics(sr, rf_val, "val", sig.mean())
        if np.isfinite(m["Sharpe"]) and m["Sharpe"] > best_sharpe:
            best_sharpe, best_t = m["Sharpe"], t

    # --- τελική αποτίμηση στο test, μία φορά, με το επιλεγμένο threshold ---
    strat_r, sig, n_trades = run_signal(test_pred, s["yte"], rf_test, best_t)
    label = f"{row['feature_set']}/{row['model']} (t={best_t})"
    metrics = perf_metrics(strat_r, rf_test, label, sig.mean(), n_trades)
    return metrics, s["yte"], rf_test, s["dte"]

def permutation_test_sharpe(y_actual, rf, exposure_frac, observed_sharpe,
                            n_perm=2000, seed=42):
    """
    Null hypothesis: η χρονική στιγμή της έκθεσης δεν έχει σημασία — μόνο
    το ΠΟΣΟΣΤΟ έκθεσης. Αναδιατάσσει τυχαία το ΠΌΤΕ είσαι μέσα/έξω, κρατώντας
    σταθερό το πόσες μέρες συνολικά (exposure_frac), και μετρά πόσο συχνά
    ένα τυχαίο timing θα έδινε Sharpe >= του παρατηρούμενου.
    """
    rs = np.random.RandomState(seed)
    n = len(y_actual)
    n_exposed = int(round(exposure_frac * n))

    sharpes = []
    for _ in range(n_perm):
        signal = np.zeros(n)
        signal[rs.choice(n, n_exposed, replace=False)] = 1.0
        r = signal * y_actual + (1 - signal) * rf
        excess = r - rf
        s = excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS) if excess.std(ddof=1) > 0 else np.nan
        sharpes.append(s)

    sharpes = np.array(sharpes)
    sharpes = sharpes[np.isfinite(sharpes)]
    p_value = (sharpes >= observed_sharpe).mean()
    return p_value, sharpes


def run_backtest():
    winners = pd.read_csv(SELECTED_PATH)
    top3 = winners.sort_values("val_bal_acc_mean", ascending=False).head(N_MODELS_TO_TEST)
    print("Μοντέλα προς backtest (top-3 σε validation):")
    print(top3[["feature_set", "model", "val_bal_acc_mean"]].to_string(index=False))

    market = load_snapshot()
    base = build_features(market).dropna(subset=FEATURES_TECH + [TARGET]).reset_index(drop=True)
    rf_map = base.set_index("Date")["RiskFreeDaily"]

    all_splits = prepare_data()
    splits_cache = {}
    for _, row in top3.iterrows():
        key = (row["feature_set"], int(row["lookback"]))
        if key not in splits_cache:
            splits_cache[key] = get_sequences(all_splits[row["feature_set"]],
                                              FEATURE_SETS[row["feature_set"]], key[1])

    rows = []
    for _, row in top3.iterrows():
        print(f"\nBacktesting {row['feature_set']}/{row['model']}...")
        metrics, y_te, rf_test, dates = backtest_one(row, splits_cache, rf_map)

        bh = perf_metrics(y_te, rf_test, f"Buy & Hold [{row['feature_set']}/{row['model']} window]",
                          1.0, 1)
        rf_only = perf_metrics(rf_test, rf_test,
                               f"Risk-free [{row['feature_set']}/{row['model']} window]", 0.0, 0)

        rows.append(metrics)
        p_val, null_dist = permutation_test_sharpe(
            y_te, rf_test, metrics["exposure"] / 100, metrics["Sharpe"]
        )
        verdict = "ΣΗΜΑΝΤΙΚΟ" if p_val < 0.05 else "μη σημαντικό (τυχαίο timing θα το εξηγούσε)"
        print(f" Permutation test: observed Sharpe={metrics['Sharpe']:.3f},"
              f"null mean={null_dist.mean():.3f}, p={p_val:.4f} -> {verdict}")
        rows.append(bh)
        rows.append(rf_only)
        print(f"  window: {pd.to_datetime(dates).min().date()} -> {pd.to_datetime(dates).max().date()}")

    bt = pd.DataFrame(rows)
    for c in ["total_return", "CAGR", "ann_vol", "max_drawdown", "exposure"]:
        bt[c] = (bt[c] * 100).round(2)
    for c in ["Sharpe", "Sortino"]:
        bt[c] = bt[c].round(3)

    bt.to_csv(RESULTS_PATH, index=False)
    print(f"\n=== BACKTEST ({pd.to_datetime(dates).min().date()} -> "
          f"{pd.to_datetime(dates).max().date()}) ===")
    print("\nΣημείωση: κάθε μοντέλο έχει το δικό του test window (μήκος εξαρτάται "
         "από το lookback) — το buy & hold/risk-free δίπλα σε κάθε μοντέλο "
         "αντιστοιχεί ακριβώς σε εκείνο το διάστημα, όχι σε κοινό παράθυρο.")
    print(bt.to_string(index=False))
    print(f"\nΑποθηκεύτηκε: {RESULTS_PATH}")
    return bt


if __name__ == "__main__":
    run_backtest()