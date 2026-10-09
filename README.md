# Πρόβλεψη Δείκτη S&P 500 με Βαθιά Μάθηση

Πτυχιακή εργασία: εφαρμογή βαθιάς μάθησης στη μελέτη, ανάλυση και πρόβλεψη
του χρηματιστηριακού δείκτη S&P 500, με σύγκριση αρχιτεκτονικών LSTM/GRU
έναντι κλασικών μοντέλων, αυστηρή στατιστική αξιολόγηση, και αυτοματοποιημένη
εκτέλεση σε paper trading μέσω του Alpaca API.

## Κεντρικό εύρημα

Σε 10 διαμορφώσεις (5 μοντέλα × 2 σύνολα χαρακτηριστικών), **καμία δεν
διακρίνεται στατιστικά από την τυχαία πρόβλεψη** της κατεύθυνσης της
επόμενης ημέρας:

1. **Ταξινόμηση** — τα bootstrap 95% CI της balanced accuracy περιλαμβάνουν
   το 50% και στα 10 μοντέλα, ακόμη και πριν τη διόρθωση Bonferroni.
2. **Χρονισμός** — permutation test στο Sharpe ratio (σταθερό ποσοστό
   έκθεσης): καμία στρατηγική με p < 0.05, ούτε στο test ούτε στα 10
   walk-forward folds.
3. **Walk-forward** — πέντε διαδοχικά παράθυρα (1993–2026, με την κρίση του
   2008 και το 2020): balanced accuracy 51.25% ± 1.15 (LSTM) και
   50.84% ± 1.38 (GRU)· το buy & hold υπερέχει σε Sharpe σε 7/10 περιπτώσεις.

Επιπλέον, ο έλεγχος McNemar δεν δείχνει διαφορά μεταξύ των δύο συνόλων
χαρακτηριστικών (όλα p > 0.05), και τα μοντέλα εμφανίζουν mode collapse
(σ_pred/σ_actual = 0.09–0.30· έως 99.8% προβλέψεις "πάνω" στους classifiers).

Τα αποτελέσματα είναι **συνεπή** με την ασθενή μορφή της Υπόθεσης
Αποτελεσματικών Αγορών για ημερήσια πρόβλεψη του SPY. Δεν αποκλείουν μικρό
πλεονέκτημα: με ~440 ημέρες ελέγχου, διαφορές μικρότερες από ~4–5 μονάδες
δεν είναι ανιχνεύσιμες.

## Δομή

```
src/
├── data_collection.py           SPY + VIX + αποδόσεις, παγωμένο snapshot
├── features.py                  Τεχνικοί δείκτες, macro features
├── alpaca_news.py               Λήψη ιστορικών ειδήσεων (2015-2026)
├── score_alpaca_sentiment.py    FinBERT scoring, 83k άρθρα
├── alpaca_sentiment.py          Ημερήσια συγκέντρωση sentiment
├── sequences.py                 Χρονολογικό split, sliding windows, scaling
├── models.py                    LSTM / GRU αρχιτεκτονικές
├── train.py                     Training loop, πλήρης ντετερμινισμός
├── hyperparameter_search.py     Random search (RNN), grid (κλασικά)· επιλογή μόνο σε validation
├── evaluate.py                  Bootstrap CI, McNemar, Bonferroni
├── backtest.py                  Risk-adjusted metrics, permutation test
├── walk_forward.py              Επικύρωση σε 5 καθεστώτα αγοράς
└── persist_model.py             Αποθήκευση μοντέλου + scaler

trading/
├── alpaca_client.py             REST client, paper-only με ρητό έλεγχο
├── live_features.py             Κατασκευή features σε πραγματικό χρόνο
├── signal_generator.py          Φόρτωση μοντέλου, παραγωγή σήματος
└── execute.py                   Αποστολή εντολών, dry-run by default
```



## Μεθοδολογία

- **Χρονολογικός διαχωρισμός** 70/15/15 — καμία τυχαία διαίρεση.
- **Επιλογή υπερπαραμέτρων αποκλειστικά σε validation**· το test set
  αγγίζεται μία φορά. *Πρώιμη έκδοση της έρευνας επέλεγε σε test set· η
  διόρθωση αποκάλυψε ~2.4 μονάδες θετική μεροληψία.*
- **Πλήρης ντετερμινισμός**: fixed seeds, `cudnn.deterministic`, `n_jobs=1`.
- **StandardScaler** προσαρμοσμένος αποκλειστικά στο train — ο ίδιος scaler
  αποθηκεύεται και χρησιμοποιείται στο live σύστημα.
- **Backtest**: εκτέλεση στο κλείσιμο, 0.1% ανά αλλαγή θέσης, cash σε
  πραγματικό risk-free (3M T-bill).

## Δύο πηγές sentiment

Αξιολογήθηκαν δύο ανεξάρτητες πηγές χρηματοοικονομικών ειδήσεων:

| Πηγή          | Κάλυψη    | Άρθρα  | Κενές ημέρες         | Live |
|---------------|-----------|--------|----------------------|------|
| Kaggle        | 2008–2024 | 19.127 | 13.8% (MNAR p<0.001) | όχι  |
| Alpaca        | 2015–2026 | 83.669 | 0.2%                 | ναι  |

**Οι δύο πηγές δεν είναι εναλλάξιμες**: Pearson r = 0.074 στις 2.234 κοινές
ημέρες, KS test p = 3×10⁻⁴⁴. Δύο feeds που και τα δύο περιγράφονται ως
"sentiment από χρηματοοικονομικές ειδήσεις" μετρούν ουσιαστικά διαφορετικά
πράγματα — σημαντική προειδοποίηση για παρόμοια συστήματα.

## Live trading

Το σύστημα υποστηρίζει **δύο μοντέλα** παράλληλα:

- **Ενεργό** (`full10_alpaca`) — παράγει σήμα και εκτελεί εντολές
- **Shadow** (`tech7`) — παράγει σήμα, καταγράφεται, δεν εκτελεί

Η εναλλαγή γίνεται με `--feature-set` ή αλλαγή του `ACTIVE_FEATURE_SET`.

```bash
python src/persist_model.py --feature-set full10_alpaca   # εκπαίδευση
python trading/execute.py                                  # dry-run
python trading/execute.py --live                           # πραγματική εντολή
```

Ασφάλεια: dry-run από προεπιλογή, ρητός έλεγχος ότι το endpoint είναι
`paper-api`, έλεγχος ωραρίου αγοράς, μόνο long/cash χωρίς μόχλευση, πλήρες
audit trail σε `signal_log.csv` και `trade_log.csv`.

## Περιορισμοί (δηλωμένοι)

- Το συναίσθημα συγκεντρώνεται σε ημερήσιο μέσο όρο· η χρονοσήμανση των
  ειδήσεων του Alpaca δεν αξιοποιείται ενδοημερήσια.
- Macro features και sentiment προστέθηκαν μαζί· δεν έγινε ablation.
- Walk-forward μόνο στο `tech7`: η πηγή ειδήσεων ξεκινά το 2015.
- Το test window του `full10_alpaca` (~440 ημέρες, κυρίως ανοδική αγορά)
  έχει περιορισμένη στατιστική ισχύ.
- 10 τυχαίοι συνδυασμοί ανά αναδρομική αρχιτεκτονική· σταθερές
  υπερπαράμετροι στο walk-forward.
- Το scheduling εξαρτάται από τοπικό μηχάνημα· cloud deployment θα έδινε
  αδιάλειπτη λειτουργία.
## Αναπαραγωγή

```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

python src/data_collection.py
python src/alpaca_news.py
python src/score_alpaca_sentiment.py
python src/hyperparameter_search.py
python src/evaluate.py
python src/backtest.py
python src/walk_forward.py
```

Απαιτείται αρχείο `.env` με `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
`ALPACA_BASE_URL`.

## Τεχνολογίες

Python, PyTorch, scikit-learn, XGBoost, statsmodels, pandas, yfinance,
transformers (FinBERT), Alpaca Trading API.