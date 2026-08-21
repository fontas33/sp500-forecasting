"""
Χρονολογικός διαχωρισμός και sliding windows.

Καμία τυχαία διαίρεση. Ο scaler προσαρμόζεται αποκλειστικά στο train split,
ώστε να μην υπάρχει διαρροή πληροφορίας από validation/test.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def chronological_split(data, train_frac=0.70, val_frac=0.15):
    """
    Διαχωρίζει σε train/val/test με αυστηρή χρονική σειρά.

    Το data πρέπει να είναι ήδη ταξινομημένο κατά ημερομηνία (το
    build_features() το εγγυάται).
    """
    n = len(data)
    i_train_end = int(n * train_frac)
    i_val_end = int(n * (train_frac + val_frac))

    train = data.iloc[:i_train_end].reset_index(drop=True)
    val = data.iloc[i_train_end:i_val_end].reset_index(drop=True)
    test = data.iloc[i_val_end:].reset_index(drop=True)

    return train, val, test


def make_sequences(frame, feature_cols, target_col, lookback):
    """
    Μετατρέπει ένα DataFrame σε sliding windows για RNN.

    Επιστρέφει:
        X: (samples, lookback, n_features)
        y: (samples,) — ο στόχος στο τέλος κάθε παραθύρου
        dates: (samples,) — η ημερομηνία που αντιστοιχεί σε κάθε y
    """
    X_src = frame[feature_cols].values
    y_src = frame[target_col].values

    X = np.stack([X_src[i - lookback:i] for i in range(lookback, len(frame))])
    y = y_src[lookback:]
    dates = frame["Date"].values[lookback:]

    return X, y, dates


def scale_splits(X_train, X_val, X_test):
    """
    Προσαρμόζει StandardScaler ΜΟΝΟ στο train, εφαρμόζει παντού.

    Κρίσιμο σημείο διόρθωσης στην έρευνα: ο scaler δεν πρέπει
    ποτέ να "δει" val/test δεδομένα κατά το fit.
    """
    n_features = X_train.shape[2]
    scaler = StandardScaler().fit(X_train.reshape(-1, n_features))

    def transform(X):
        return scaler.transform(X.reshape(-1, n_features)).reshape(X.shape)

    return transform(X_train), transform(X_val), transform(X_test), scaler


if __name__ == "__main__":
    import pandas as pd
    from data_collection import load_snapshot
    from features import build_features, FEATURES_TECH, TARGET

    market = load_snapshot()
    data = build_features(market).dropna(subset=FEATURES_TECH + [TARGET]).reset_index(drop=True)

    train, val, test = chronological_split(data)
    print(f"Train: {len(train)}  ({train['Date'].min().date()} - {train['Date'].max().date()})")
    print(f"Val:   {len(val)}  ({val['Date'].min().date()} - {val['Date'].max().date()})")
    print(f"Test:  {len(test)}  ({test['Date'].min().date()} - {test['Date'].max().date()})")

    X_tr, y_tr, d_tr = make_sequences(train, FEATURES_TECH, TARGET, lookback=60)
    X_va, y_va, d_va = make_sequences(val, FEATURES_TECH, TARGET, lookback=60)
    X_te, y_te, d_te = make_sequences(test, FEATURES_TECH, TARGET, lookback=60)
    print(f"\nX_train shape: {X_tr.shape}")

    X_tr_s, X_va_s, X_te_s, scaler = scale_splits(X_tr, X_va, X_te)
    print(f"Μέσος όρος μετά scaling (train): {X_tr_s.mean():.6f}")
    print(f"Τυπική απόκλιση μετά scaling (train): {X_tr_s.std():.6f}")