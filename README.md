# Πρόβλεψη Δείκτη S&P 500 με Βαθιά Μάθηση

Πτυχιακή εργασία: Εφαρμογή βαθιάς μάθησης στη μελέτη, ανάλυση και πρόβλεψη
του χρηματιστηριακού δείκτη S&P 500, με σύγκριση αρχιτεκτονικών LSTM/GRU
έναντι κλασικών μοντέλων μηχανικής μάθησης, και αξιολόγηση μέσω αυστηρού
statistical backtesting.

## Κεντρικό εύρημα

Σε 10 μοντέλα (LSTM, GRU, Logistic Regression, Random Forest, XGBoost) ×
2 σύνολα χαρακτηριστικών (τεχνικοί δείκτες· τεχνικοί δείκτες + macro +
sentiment ειδήσεων), **καμία αρχιτεκτονική δεν ξεπερνά στατιστικά την
τυχαία πρόβλεψη** στην πρόβλεψη κατεύθυνσης επόμενης ημέρας, ακόμη και μετά από Bonferroni correction για πολλαπλές συγκρίσεις. Το εύρημα επιβεβαιώνεται σε
τρία ανεξάρτητα επίπεδα ανάλυσης:

1. **Ταξινόμηση** — bootstrap 95% CI για balanced accuracy περιλαμβάνει
  το 50% σε 9/10 μοντέλα· το μοναδικό «σημαντικό» αποτέλεσμα δεν
  επιβιώνει τη διόρθωση Bonferroni.
2. **Κατανομή προβλέψεων** — τα recurrent δίκτυα συρρικνώνονται προς
  σχεδόν-σταθερή τιμή (σ(pred)/σ(actual) έως 0.065)· οι classifiers
  μεροληπτούν συστηματικά προς μία κλάση (έως 98.7% προβλέψεις «πάνω»).
3. **Οικονομική απόδοση** — ένα φαινομενικά ισχυρό backtest αποτέλεσμα
  (Sharpe 0.315) αποδεικνύεται μη διακριτό από τυχαίο χρονισμό θέσης
  (permutation test, p=0.375).

## Δομή

src/
├── data_collection.py # Λήψη SPY + VIX + αποδόσεις (παγωμένο snapshot)
├── features.py # Τεχνικοί δείκτες, macro features
├── sentiment.py # FinBERT sentiment (cached, βλ. σημείωση)
├── sequences.py # Χρονολογικό split, sliding windows, scaling
├── models.py # LSTM / GRU αρχιτεκτονικές
├── train.py # Training loop με πλήρη ντετερμινισμό
├── hyperparameter_search.py # Random search — επιλογή ΜΟΝΟ σε validation
├── evaluate.py # Τελική test αποτίμηση, bootstrap CI, McNemar
└── backtest.py # Risk-adjusted backtest, permutation test

data/
├── market_snapshot.csv # Παγωμένα δεδομένα (αναπαράγεται με data_collection.py)
├── news_sentiment_scored.csv # FinBERT scores (υπολογίστηκε σε Kaggle GPU)
├── validation_search_log.csv # Πλήρες log του hyperparameter search
├── selected_configs.csv # Νικητές ανά (feature_set, model)
├── final_test_results.csv # Τελικά test metrics
└── backtest_results.csv # Backtest metrics


## Μεθοδολογία

- **Χρονολογικός διαχωρισμός** 70/15/15 (train/val/test) — καμία τυχαία
 διαίρεση.
- **Επιλογή υπερπαραμέτρων αποκλειστικά σε validation set** — το test
 set αγγίζεται μία μόνο φορά, στο τέλος. (Πρώιμη έκδοση της έρευνας
 επέλεγε σε test set· η διόρθωση αποκάλυψε ~2.4 μονάδες θετική
 μεροληψία στα προηγούμενα αποτελέσματα.)
- **Πλήρης ντετερμινισμός**: fixed seeds, `cudnn.deterministic=True`,
 `n_jobs=1` σε όλα τα sklearn/xgboost μοντέλα.
- **StandardScaler** προσαρμοσμένος αποκλειστικά στο train split.
- **Στατιστικός έλεγχος**: bootstrap 95% CI, McNemar test, διόρθωση
 Bonferroni για πολλαπλές συγκρίσεις, permutation test για Sharpe.
- **Backtest**: εκτέλεση στο κλείσιμο, κόστος 0.1%/συναλλαγή, cash σε
 πραγματικό risk-free (3M T-bill) — όχι 0%.

## Περιορισμοί (δηλωμένοι)

- Το dataset ειδήσεων δεν παρέχει ώρα δημοσίευσης εντός ημέρας.
- 13.8% των ημερών χωρίς κάλυψη ειδήσεων συμπληρώθηκαν με forward-fill·
 έλεγχος Mann-Whitney έδειξε ότι η απουσία **δεν είναι τυχαία** (MNAR,
 p<0.001) — οι ημέρες χωρίς είδηση είναι συστηματικά πιο ασταθείς.
- Ένα μόνο test regime (κυρίως ανοδική περίοδος 2021-2024).
- Random search αντί εξαντλητικού grid search.

## Αναπαραγωγή

```bash
python -m venv venv
venv\Scripts\activate.bat        # Windows
pip install -r requirements.txt

python src/data_collection.py     # Λήψη δεδομένων (παραλείπεται αν υπάρχει ήδη snapshot)
python src/hyperparameter_search.py
python src/evaluate.py
python src/backtest.py
```

Το sentiment scoring (FinBERT) απαιτεί το πρωτότυπο dataset ειδήσεων
(Kaggle: `dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024`)
και τρέχει καλύτερα σε GPU· το ήδη υπολογισμένο `news_sentiment_scored.csv`
περιλαμβάνεται στο repo.

## Τεχνολογίες

Python, PyTorch, scikit-learn, XGBoost, statsmodels, pandas, yfinance,
transformers (FinBERT), Alpaca API (σε εξέλιξη — αυτόματη εκτέλεση
paper-trading).