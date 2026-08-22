"""
Αρχιτεκτονικές νευρωνικών δικτύων για πρόβλεψη log-return.
LSTM και GRU με ίδια δομή (ίδιο hidden size, layers, dropout, 
τελικό Linear layer), ώστε η σύγκριση μεταξύ τους να είναι δίκαιη - μόνο ο
τύπος του recurrent layer διαφέρει.
"""

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            n_features, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(
            n_features, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


ARCHITECTURES = {"LSTM": LSTMModel, "GRU": GRUModel}


if __name__ == "__main__":
    import torch

    for name, Model in ARCHITECTURES.items():
        model = Model(n_features=7)
        dummy = torch.randn(4, 60, 7)  # (batch, lookback, features)
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output shape={tuple(out.shape)}, παράμετροι={n_params:,}")