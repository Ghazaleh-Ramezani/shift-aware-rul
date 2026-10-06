"""Hybrid CNN + LSTM regressor and its training loop."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class HybridCNNLSTM(nn.Module):
    """Parallel CNN (local patterns) and LSTM (temporal trend) paths, fused.

    The input BatchNorm standardizes the sensors inside the network, so
    re-estimating its statistics on target data (AdaBN) re-centres the inputs.
    """

    def __init__(self, n_features: int, conv_channels: int = 32, lstm_hidden: int = 32,
                 fusion_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, 5, padding=2),
            nn.BatchNorm1d(conv_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, 3, padding=1),
            nn.BatchNorm1d(conv_channels), nn.ReLU(),
        )
        self.lstm = nn.LSTM(n_features, lstm_hidden, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(conv_channels + lstm_hidden, fusion_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(fusion_dim, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_bn(x.transpose(1, 2))          # (B, F, T)
        c = self.conv(z).mean(dim=2)                  # (B, C)
        h, _ = self.lstm(z.transpose(1, 2))           # (B, T, H)
        return self.fusion(torch.cat([c, h[:, -1]], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, y_scale: float, batch_size: int = 2048) -> np.ndarray:
    out = [model(torch.from_numpy(x[i:i + batch_size])) for i in range(0, len(x), batch_size)]
    # RUL labels live in [0, cap]; clip so a far-out-of-distribution input cannot
    # produce physically meaningless predictions.
    return np.clip(torch.cat(out).numpy() * y_scale, 0.0, y_scale)


@torch.no_grad()
def embed(model: HybridCNNLSTM, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out = [model.embed(torch.from_numpy(x[i:i + batch_size])) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()


def train_model(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *,
                y_scale: float, seed: int, epochs: int = 30, batch_size: int = 256,
                lr: float = 1e-3, weight_decay: float = 1e-5, patience: int = 6,
                model_kwargs: dict | None = None, log=print) -> HybridCNNLSTM:
    """Train with MSE on y / y_scale; keep the weights with the best validation RMSE."""
    set_seed(seed)
    model = HybridCNNLSTM(x.shape[2], **(model_kwargs or {}))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y / y_scale)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            if len(b) < 2:  # BatchNorm needs more than one sample
                continue
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        val_rmse = float(np.sqrt(np.mean((predict(model, x_val, y_scale) - y_val) ** 2)))
        log(f"  seed {seed} epoch {epoch + 1:02d} val RMSE {val_rmse:.2f}")
        if val_rmse < best - 1e-3:
            best, best_state, bad = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model
