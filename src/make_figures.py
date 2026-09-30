"""
Παραγωγή γραφημάτων για την πτυχιακή εργασία.
 
Δύο λειτουργίες:
  --quick   Μόνο γραφήματα που προκύπτουν από τα αποθηκευμένα CSV (παίρνει δευτερόλεπτα).
  (πλήρης)  Επιπλέον ξανατρέχει τα μοντέλα για γραφήματα που απαιτούν
            προβλέψεις σε επίπεδο δείγματος (κατανομές, συσσωρευτική απόδοση).
 
Όλα τα γραφήματα αποθηκεύονται στο figures/ σε 300 dpi, σε μορφή κατάλληλη
για εισαγωγή σε έγγραφο Word.
 
Λειτουργεί εξ ολοκλήρου εκτός σύνδεσης — διαβάζει το παγωμένο snapshot.
"""
 
import argparse
import warnings
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore")
 
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
 
# --- Ενιαίο ύφος: ουδέτερο, για ασπρόμαυρη εκτύπωση ---
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})
 
NAVY, TEAL, RED, GREY = "#1f3864", "#1baf7a", "#c0392b", "#8c8c8c"
 
 
def save(fig, name, caption):
    path = FIGS / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name:38s} — {caption}")
 
 
def read(name):
    p = DATA / name
    if not p.exists():
        print(f"  [παράλειψη] δεν βρέθηκε {name}")
        return None
    return pd.read_csv(p)
 
 
# ═══════════════════════════════════════════════════════════════
#  ΜΕΡΟΣ Α — Γραφήματα από αποθηκευμένα CSV
# ═══════════════════════════════════════════════════════════════
 
def fig_price_history():
    """Σχήμα: ιστορικό τιμής SPY με σημειωμένα τα καθεστώτα αγοράς."""
    df = read("market_snapshot.csv")
    if df is None:
        return
    df["Date"] = pd.to_datetime(df["Date"])
 
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
 
    ax[0].plot(df["Date"], df["Close"], color=NAVY, lw=1.1)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Τιμή κλεισίματος (USD, λογαριθμική κλίμακα)")
    ax[0].set_title("Ιστορική εξέλιξη του SPY και της μεταβλητότητας της αγοράς")
 
    for start, end, label in [("2007-10-01", "2009-03-31", "Κρίση 2008"),
                              ("2020-02-15", "2020-04-15", "COVID-19"),
                              ("2022-01-01", "2022-10-15", "Πτώση 2022")]:
        ax[0].axvspan(pd.Timestamp(start), pd.Timestamp(end), color=RED, alpha=0.12)
        ax[1].axvspan(pd.Timestamp(start), pd.Timestamp(end), color=RED, alpha=0.12)
        ax[0].text(pd.Timestamp(start), df["Close"].max() * 0.95, f" {label}",
                   fontsize=7.5, color=RED, va="top")
 
    if "VIX" in df.columns:
        ax[1].plot(df["Date"], df["VIX"], color=GREY, lw=0.8)
        ax[1].set_ylabel("Δείκτης VIX")
    ax[1].set_xlabel("Έτος")
 
    save(fig, "01_istoriko_timis",
         "Ιστορικό τιμής SPY & VIX (Κεφ. 4 — δεδομένα)")
 
 
def fig_model_comparison():
    """Σχήμα: ισορροπημένη ακρίβεια ανά μοντέλο, με ράβδους σφάλματος."""
    df = read("final_test_results.csv")
    if df is None:
        return
    df = df.sort_values("test_bal_acc_mean")
    labels = [f"{r.feature_set}\n{r.model}" for r in df.itertuples()]
    y = np.arange(len(df))
 
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = [TEAL if v > 0.5 else RED for v in df["test_bal_acc_mean"]]
    ax.barh(y, df["test_bal_acc_mean"] * 100,
            xerr=df["test_bal_acc_std"] * 100,
            color=colors, alpha=0.75, height=0.65,
            error_kw={"ecolor": "#444", "capsize": 3, "lw": 1})
    ax.axvline(50, color="k", ls="--", lw=1.2)
    ax.text(50.15, len(df) - 0.4, "Τυχαία πρόβλεψη (50%)", fontsize=8, rotation=90,
            va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Ισορροπημένη ακρίβεια στο σύνολο ελέγχου (%)")
    ax.set_title("Σύγκριση μοντέλων — μέσος όρος 5 seeds")
    lo = min(45, df["test_bal_acc_mean"].min() * 100 - 2)
    ax.set_xlim(lo, max(56, df["test_bal_acc_mean"].max() * 100 + 2))
 
    save(fig, "02_sygkrisi_montelon",
         "Ισορροπημένη ακρίβεια ανά μοντέλο (Κεφ. 6.1)")
 
 
def fig_val_vs_test():
    """Σχήμα-κλειδί: η κατάταξη στο validation δεν προβλέπει το test."""
    df = read("final_test_results.csv")
    if df is None:
        return
 
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    df = df.sort_values("test_bal_acc_mean", ascending=False).reset_index(drop=True)
    for i, r in enumerate(df.itertuples(), 1):
        m = "o" if r.model in ("LSTM", "GRU") else "s"
        x, yv = r.val_bal_acc * 100, r.test_bal_acc_mean * 100
        ax.scatter(x, yv, s=180, marker=m, color=NAVY, alpha=0.8, zorder=3)
        ax.annotate(str(i), (x, yv), color="white", fontsize=7.5,
                    ha="center", va="center", zorder=4, fontweight="bold")
 
    lims = [min(df.val_bal_acc.min(), df.test_bal_acc_mean.min()) * 100 - 1.5,
            max(df.val_bal_acc.max(), df.test_bal_acc_mean.max()) * 100 + 1.5]
    ax.plot(lims, lims, color=GREY, ls=":", lw=1.2, label="Τέλεια αντιστοιχία")
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.axvline(50, color="k", ls="--", lw=1)
 
    r_s = df["val_bal_acc"].corr(df["test_bal_acc_mean"], method="spearman")
    ax.set_xlabel("Ισορροπημένη ακρίβεια — σύνολο επικύρωσης (%)")
    ax.set_ylabel("Ισορροπημένη ακρίβεια — σύνολο ελέγχου (%)")
    ax.set_title(f"Επικύρωση έναντι ελέγχου (Spearman ρ = {r_s:.2f})")
    ax.set_xlim(lims); ax.set_ylim(lims)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls=":", color=GREY, label="Τέλεια αντιστοιχία"),
               Line2D([], [], ls="", marker="o", color=NAVY, label="Αναδρομικά δίκτυα"),
               Line2D([], [], ls="", marker="s", color=NAVY, label="Κλασικά μοντέλα")]
    leg1 = ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.add_artist(leg1)

    key = [Line2D([], [], ls="", label=f"{i}. {r.feature_set}/{r.model}")
           for i, r in enumerate(df.itertuples(), 1)]
    ax.legend(handles=key, fontsize=7, loc="upper left", handlelength=0,
              handletextpad=0, labelspacing=0.32, title="Μοντέλα",
              title_fontsize=7.5)
 
    save(fig, "03_validation_vs_test",
         "Απουσία αντιστοιχίας validation–test (Κεφ. 6.1 / 7.1)")
 
 
def fig_mode_collapse_bias():
    """Σχήμα: μεροληψία προς την πλειοψηφική κλάση ανά μοντέλο."""
    df = read("final_test_results.csv")
    if df is None or "pct_up_mean" not in df.columns:
        return
    df = df.sort_values("pct_up_mean")
    labels = [f"{r.feature_set}/{r.model}" for r in df.itertuples()]
    y = np.arange(len(df))
 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y, df["pct_up_mean"] * 100, color=NAVY, alpha=0.7, height=0.6)
    snap = read("market_snapshot.csv")
    up = (np.log(snap["Close"]).diff().dropna() > 0).mean() * 100
    ax.axvline(up, color=RED, ls="--", lw=1.3)
    ax.text(up + 0.8, -0.8, "Ποσοστό ανοδικών ημερών\n(όλη η περίοδος, "
            + f"{up:.1f}".replace(".", ",") + "%)", fontsize=7.5, color=RED)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Ποσοστό προβλέψεων «άνοδος» (%)")
    ax.set_title("Μεροληψία προβλέψεων προς την πλειοψηφική κλάση")
    ax.set_xlim(0, 105)
 
    save(fig, "04_meroleipsia_klasis",
         "Μεροληψία κλάσης — ένδειξη κατάρρευσης (Κεφ. 6.4)")
 
 
def fig_walkforward():
    """Σχήμα: ακρίβεια και Sharpe ανά χρονικό παράθυρο."""
    df = read("walkforward_results.csv")
    if df is None:
        return
 
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    archs = df["arch"].unique()
    w = 0.36
    x = np.arange(df["fold"].nunique())
 
    for i, a in enumerate(archs):
        s = df[df["arch"] == a].sort_values("fold")
        ax[0].bar(x + (i - 0.5) * w, s["bal_acc_mean"] * 100, w,
                  yerr=s["bal_acc_std"] * 100, label=a,
                  color=[NAVY, TEAL][i % 2], alpha=0.8,
                  error_kw={"ecolor": "#444", "capsize": 2.5, "lw": 0.9})
    ax[0].axhline(50, color="k", ls="--", lw=1.2)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"Fold {i+1}" for i in x], fontsize=8)
    ax[0].set_ylabel("Ισορροπημένη ακρίβεια (%)")
    ax[0].set_title("Ακρίβεια ανά χρονικό παράθυρο")
    ax[0].set_ylim(45, 56)
    ax[0].legend(fontsize=8)
 
    for i, a in enumerate(archs):
        s = df[df["arch"] == a].sort_values("fold")
        ax[1].bar(x + (i - 0.5) * w, s["strat_sharpe"], w, label=f"Στρατηγική {a}",
                  color=[NAVY, TEAL][i % 2], alpha=0.8)
    bh = df.groupby("fold")["bh_sharpe"].first().sort_index()
    ax[1].plot(x, bh.values, "D--", color=RED, ms=6, lw=1.4,
               label="Αγορά & διακράτηση")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"Fold {i+1}" for i in x], fontsize=8)
    ax[1].set_ylabel("Δείκτης Sharpe (ετησιοποιημένος)")
    ax[1].set_title("Απόδοση προσαρμοσμένη στον κίνδυνο")
    ax[1].legend(fontsize=8)
 
    save(fig, "05_walkforward",
         "Επικύρωση walk-forward σε 5 καθεστώτα (Κεφ. 6.6)")
 
 
def fig_sentiment_sources():
    """Σχήμα: σύγκριση των δύο πηγών συναισθήματος."""
    alp = read("alpaca_sentiment_daily.csv")
    kag = read("news_sentiment_scored.csv")
    if alp is None:
        return
    alp["Date"] = pd.to_datetime(alp["Date"])
 
    if kag is not None and "Sentiment" in kag.columns:
        kag["Date"] = pd.to_datetime(kag["Date"])
        sign = kag["Sentiment"].map({"positive": 1.0, "negative": -1.0, "neutral": 0.0})
        kag["num"] = sign * kag["SentimentScore"]
        kd = kag.groupby("Date")["num"].mean().reset_index(name="kaggle")
        m = kd.merge(alp[["Date", "SentimentMean"]], on="Date", how="inner")
    else:
        m = pd.DataFrame()
 
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
 
    ax[0].hist(alp["SentimentMean"].dropna(), bins=60, color=NAVY, alpha=0.7,
               label="Alpaca news API", density=True)
    if len(m):
        ax[0].hist(m["kaggle"].dropna(), bins=60, color=RED, alpha=0.5,
                   label="Δημόσιο σύνολο", density=True)
    ax[0].axvline(0, color="k", ls="--", lw=1)
    ax[0].set_xlabel("Ημερήσιος μέσος όρος συναισθήματος")
    ax[0].set_ylabel("Πυκνότητα")
    ax[0].set_title("Κατανομές των δύο πηγών")
    ax[0].legend(fontsize=8)
 
    if len(m):
        r = m["kaggle"].corr(m["SentimentMean"])
        ax[1].scatter(m["kaggle"], m["SentimentMean"], s=7, alpha=0.3, color=NAVY)
        ax[1].axhline(0, color="k", lw=0.8); ax[1].axvline(0, color="k", lw=0.8)
        ax[1].set_xlabel("Δημόσιο σύνολο δεδομένων")
        ax[1].set_ylabel("Alpaca news API")
        ax[1].set_title(f"Συσχέτιση στις κοινές ημέρες (r = {r:.3f}, n = {len(m)})")
    else:
        ax[1].text(0.5, 0.5, "Δεν βρέθηκε το δεύτερο σύνολο\nγια σύγκριση",
                   ha="center", va="center", transform=ax[1].transAxes, color=GREY)
        ax[1].set_axis_off()
 
    save(fig, "06_phiges_sentiment",
         "Σύγκριση πηγών συναισθήματος (Κεφ. 4.3 / 7.4)")
 
 
def fig_search_distribution():
    """Σχήμα: κατανομή επιδόσεων όλων των συνδυασμών υπερπαραμέτρων."""
    df = read("validation_search_log.csv")
    if df is None:
        return
 
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (name, g) in enumerate(df.groupby("model")):
        ax.scatter(np.full(len(g), i) + np.random.uniform(-0.16, 0.16, len(g)),
                   g["val_bal_acc_mean"] * 100, s=28, alpha=0.6, color=NAVY)
    ax.axhline(50, color=RED, ls="--", lw=1.3)
    ax.set_xticks(range(df["model"].nunique()))
    ax.set_xticklabels(sorted(df["model"].unique()), fontsize=9)
    ax.set_ylabel("Ισορροπημένη ακρίβεια — επικύρωση (%)")
    ax.set_title("Κατανομή επιδόσεων όλων των δοκιμασθέντων συνδυασμών")
 
    save(fig, "07_katanomi_anazitisis",
         "Εύρος αποτελεσμάτων αναζήτησης υπερπαραμέτρων (Κεφ. 6.1)")
 
 
# ═══════════════════════════════════════════════════════════════
#  ΜΕΡΟΣ Β — Γραφήματα που απαιτούν επανεκτέλεση μοντέλων
# ═══════════════════════════════════════════════════════════════
 
def fig_predictions_and_equity():
    """Κατανομή προβλέψεων (mode collapse) και καμπύλη συσσωρευτικής απόδοσης."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
 
    from hyperparameter_search import (FEATURE_SETS, prepare_data, get_sequences,
                                       build_flat_model, flatten)
    from evaluate import coerce_params
    from train import train_model, predict, set_all_seeds
    from backtest import perf_metrics, run_signal
    from data_collection import load_snapshot
    from features import build_features, FEATURES_TECH, TARGET
 
    winners = read("selected_configs.csv")
    if winners is None:
        return
 
    best = winners.sort_values("val_bal_acc_mean", ascending=False).iloc[0]
    fs, mn, lb = best["feature_set"], best["model"], int(best["lookback"])
    print(f"  Επανεκπαίδευση {fs}/{mn} (lookback={lb}) για τα γραφήματα...")
 
    splits = prepare_data()
    s = get_sequences(splits[fs], FEATURE_SETS[fs], lb)
 
    if mn in ("LSTM", "GRU"):
        model = train_model(mn, s["Xtr"], s["ytr"], s["Xva"], s["yva"],
                            hidden_size=int(best["hidden"]), dropout=float(best["dropout"]),
                            lr=float(best["lr"]), seed=1)
        pred = predict(model, s["Xte"])
        same_units = True
    else:
        raw = {k[2:]: best[k] for k in best.index
               if str(k).startswith("p_") and pd.notna(best[k])}
        params = coerce_params(raw)
        if mn == "RF" and "max_depth" not in params:
            params["max_depth"] = None
        set_all_seeds(1)
        clf = build_flat_model(mn, params, 1).fit(flatten(s["Xtr"]),
                                                  (s["ytr"] > 0).astype(int))
        pred = clf.predict_proba(flatten(s["Xte"]))[:, 1] - 0.5
        same_units = False
 
    y, dates = s["yte"], pd.to_datetime(s["dte"])
 
    # --- Σχήμα: κατανομή προβλέψεων έναντι πραγματικών ---
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(y, bins=70, color=GREY, alpha=0.65, label="Πραγματικές αποδόσεις",
            density=True)
    ax.hist(pred, bins=70, color=NAVY, alpha=0.75, label="Προβλέψεις μοντέλου",
            density=True)
    ax.axvline(0, color="k", ls="--", lw=1)
    ratio = pred.std() / y.std()
    extra = f"  —  σ(πρόβλ.)/σ(πραγμ.) = {ratio:.3f}" if same_units else ""
    ax.set_xlabel("Λογαριθμική απόδοση" if same_units
                  else "Τιμή εξόδου μοντέλου (διαφορετικές μονάδες)")
    ax.set_ylabel("Πυκνότητα")
    ax.set_title(f"Κατανομή προβλέψεων — {fs}/{mn}{extra}")
    ax.legend(fontsize=8)
    save(fig, "08_katanomi_provlepseon",
         "Διάγνωση κατάρρευσης προβλέψεων (Κεφ. 6.4)")
 
    # --- Σχήμα: συσσωρευτική απόδοση + drawdown ---
    market = load_snapshot()
    base = build_features(market).dropna(subset=FEATURES_TECH + [TARGET])
    rf_map = base.set_index("Date")["RiskFreeDaily"]
    rf = rf_map.reindex(dates).ffill().fillna(0).values
 
    strat, sig, n_tr = run_signal(pred, y, rf, 0.0)
    cum_s, cum_b, cum_r = (np.cumprod(1 + a) for a in (strat, y, rf))
 
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                          gridspec_kw={"height_ratios": [2.4, 1]})
    ax[0].plot(dates, cum_b, color=GREY, lw=1.5, label="Αγορά & διακράτηση")
    ax[0].plot(dates, cum_s, color=TEAL, lw=1.5,
               label=f"Στρατηγική μοντέλου ({n_tr} συναλλαγές)")
    ax[0].plot(dates, cum_r, color=RED, ls="--", lw=1.1, label="Μηδενικού κινδύνου")
    ax[0].set_ylabel("Συσσωρευτική απόδοση (×)")
    ax[0].set_title("Αναδρομικός έλεγχος — κόστος 0,1% ανά συναλλαγή, "
                    "εκτέλεση στο κλείσιμο")
    ax[0].legend(fontsize=8)
 
    for c, lab, col in [(cum_b, "Αγορά & διακράτηση", GREY),
                        (cum_s, "Στρατηγική", TEAL)]:
        dd = (c - np.maximum.accumulate(c)) / np.maximum.accumulate(c)
        ax[1].fill_between(dates, dd * 100, 0, alpha=0.35, color=col, label=lab)
    ax[1].set_ylabel("Πτώση από κορυφή (%)")
    ax[1].set_xlabel("Ημερομηνία")
    ax[1].legend(fontsize=8)
    save(fig, "09_sossoreftiki_apodosi",
         "Συσσωρευτική απόδοση & drawdown (Κεφ. 6.5)")
 
    m1 = perf_metrics(strat, rf, "strategy", sig.mean(), n_tr)
    m2 = perf_metrics(y, rf, "buyhold", 1.0, 1)
    print(f"    Sharpe στρατηγικής {m1['Sharpe']:+.3f} | "
          f"αγοράς & διακράτησης {m2['Sharpe']:+.3f}")
 
 
# ═══════════════════════════════════════════════════════════════
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Μόνο γραφήματα από CSV, χωρίς επανεκπαίδευση")
    args = ap.parse_args()
 
    print("Παραγωγή γραφημάτων\n")
    print("Από αποθηκευμένα αποτελέσματα:")
    for f in (fig_price_history, fig_model_comparison, fig_val_vs_test,
              fig_mode_collapse_bias, fig_walkforward, fig_sentiment_sources,
              fig_search_distribution):
        try:
            f()
        except Exception as exc:
            print(f"  [σφάλμα] {f.__name__}: {type(exc).__name__}: {exc}")
 
    if not args.quick:
        print("\nΜε επανεκπαίδευση μοντέλου:")
        try:
            fig_predictions_and_equity()
        except Exception as exc:
            print(f"  [σφάλμα] {type(exc).__name__}: {exc}")
    else:
        print("\n(παραλείφθηκαν τα γραφήματα που απαιτούν επανεκπαίδευση — "
              "τρέξε χωρίς --quick)")
 
    print(f"\nΤα αρχεία βρίσκονται στο {FIGS}")
 
 
if __name__ == "__main__":
    main()