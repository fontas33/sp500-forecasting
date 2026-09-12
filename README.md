# Πρόβλεψη Δείκτη S&P 500 με Βαθιά Μάθηση

Πτυχιακή εργασία: εφαρμογή βαθιάς μάθησης στη μελέτη, ανάλυση και πρόβλεψη
του χρηματιστηριακού δείκτη S&P 500, με σύγκριση αρχιτεκτονικών LSTM/GRU
έναντι κλασικών μοντέλων, αυστηρή στατιστική αξιολόγηση, και αυτοματοποιημένη
εκτέλεση σε paper trading μέσω του Alpaca API.

## Κεντρικό εύρημα

Σε 10 συνδυασμούς (5 αρχιτεκτονικές × 2 σύνολα χαρακτηριστικών), **καμία
προσέγγιση δεν ξεπερνά στατιστικά την τυχαία πρόβλεψη** στην κατεύθυνση της
επόμενης ημέρας. Το εύρημα επιβεβαιώνεται σε **τέσσερα ανεξάρτητα επίπεδα**:

1. **Ταξινόμηση** — bootstrap 95% CI περιλαμβάνει το 50% και στα 10 μοντέλα,
   ακόμη και πριν τη διόρθωση Bonferroni.
2. **Σύγκριση χαρακτηριστικών** — McNemar test, όλα p > 0.05: η προσθήκη
   macro και sentiment features δεν παράγει μετρήσιμη διαφορά.
3. **Κατανομή προβλέψεων** — mode collapse σε όλα τα μοντέλα: τα recurrent
   συρρικνώνονται προς σχεδόν-σταθερή τιμή (σ_pred/σ_actual = 0.09–0.30),
   οι classifiers μεροληπτούν προς μία κλάση (έως 99.7% προβλέψεις "πάνω").
4. **Οικονομική απόδοση** — permutation test στο Sharpe ratio: 0/10 folds
   με p < 0.05. Κάθε φαινομενική υπεραπόδοση εξηγείται από τυχαίο χρονισμό
   θέσης, όχι από προγνωστική ικανότητα.

**Walk-forward επικύρωση:** πέντε διαδοχικά χρονικά παράθυρα (1993–2026),
συμπεριλαμβανομένων της κρίσης 2008 και του COVID crash. Balanced accuracy
51.25% ± 1.15 (LSTM) και 50.84% ± 1.38 (GRU). Το buy & hold υπερέχει σε
Sharpe σε 8/10 περιπτώσεις.

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
├── hyperparameter_search.py     Random search, επιλογή μόνο σε validation
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
- **Backtest**: εκτέλεση στο κλείσιμο, 0.1%/συναλλαγή, cash σε πραγματικό
  risk-free (3M T-bill).

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

- Το dataset ειδήσεων δεν παρέχει ώρα δημοσίευσης εντός ημέρας.
- Walk-forward εφαρμόστηκε μόνο στο `tech7`: η πηγή ειδήσεων ξεκινά το 2015,
  ανεπαρκές ιστορικό για ουσιαστικά folds στο `full10_alpaca`.
- Το test window του `full10_alpaca` καλύπτει 18 μήνες έντονα ανοδικής
  αγοράς — μειωμένη στατιστική ισχύς.
- Το scheduling εξαρτάται από τοπικό μηχάνημα· cloud deployment θα έδινε
  πλήρη αξιοπιστία.
- Random search αντί εξαντλητικού grid search.

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