"""
Εκπαίδευση ενός LSTM/GRU μοντέλου, με πλήρη ντετερμινισμό.

Ταυτόσημη λογική με το επικυρωμένο Kaggle pipeline: mini-batch training,
early stopping στο validation loss, seeds παντού.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import ARCHITECTURES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed):
    """Πλήρης ντετερμινισμός — ίδιο seed πρέπει να δίνει ίδιο αποτέλεσμα."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor_dataset(X, y):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return X_t, y_t


def train_model(arch_name, X_train, y_train, X_val, y_val,
                hidden_size=64, dropout=0.2, lr=0.001,
                seed=1, max_epochs=100, patience=8, batch_size=64):
    """
    Εκπαιδεύει ένα μοντέλο με early stopping στο validation loss.

    Επιστρέφει το εκπαιδευμένο μοντέλο (με τα καλύτερα βάρη φορτωμένα,
    όχι απαραίτητα του τελευταίου epoch).
    """
    set_all_seeds(seed)

    X_tr_t, y_tr_t = to_tensor_dataset(X_train, y_train)
    X_va_t, y_va_t = to_tensor_dataset(X_val, y_val)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    Model = ARCHITECTURES[arch_name]
    model = Model(n_features=X_train.shape[2], hidden_size=hidden_size, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va_t), y_va_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, X):
    """Επιστρέφει τις προβλέψεις ως 1D numpy array."""
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_t).cpu().numpy().ravel()


if __name__ == "__main__":
    from data_collection import load_snapshot
    from features import build_features, FEATURES_TECH, TARGET
    from sequences import chronological_split, make_sequences, scale_splits

    print(f"Device: {device}")

    market = load_snapshot()
    data = build_features(market).dropna(subset=FEATURES_TECH + [TARGET]).reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(data)

    X_tr, y_tr, _ = make_sequences(train_df, FEATURES_TECH, TARGET, lookback=60)
    X_va, y_va, _ = make_sequences(val_df, FEATURES_TECH, TARGET, lookback=60)
    X_tr_s, X_va_s, _, _ = scale_splits(X_tr, X_va, X_va)  # X_va ξαναπερνά μόνο για το transform

    import time
    t0 = time.time()
    model = train_model("LSTM", X_tr_s, y_tr, X_va_s, y_va, seed=1)
    print(f"Εκπαίδευση σε {time.time()-t0:.1f}s")

    preds = predict(model, X_va_s)
    mae = np.mean(np.abs(preds - y_va))
    print(f"Validation MAE: {mae:.6f}")