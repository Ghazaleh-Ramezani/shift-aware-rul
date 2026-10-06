"""Uncertainty quantification: MC dropout, deep ensembles, a GP head, and conformal intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from torch import nn

from .models import HybridCNNLSTM, embed, predict

SIGMA_FLOOR = 0.5  # RUL cycles; keeps normalized scores finite


@dataclass
class Prediction:
    mu: np.ndarray
    sigma: np.ndarray


def _floor(sigma: np.ndarray) -> np.ndarray:
    return np.maximum(sigma, SIGMA_FLOOR)


def mc_dropout(model: nn.Module, x: np.ndarray, y_scale: float, passes: int = 30,
               seed: int = 0) -> Prediction:
    """Dropout on, BatchNorm in eval mode, `passes` stochastic forward passes."""
    torch.manual_seed(seed)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    draws = np.stack([predict(model, x, y_scale) for _ in range(passes)])
    model.eval()
    return Prediction(draws.mean(0), _floor(draws.std(0)))


def ensemble(models: list[nn.Module], x: np.ndarray, y_scale: float) -> Prediction:
    draws = np.stack([predict(m.eval(), x, y_scale) for m in models])
    return Prediction(draws.mean(0), _floor(draws.std(0)))


class GPHead:
    """Exact GP regression on the network's penultimate embeddings."""

    def __init__(self, n_max: int = 1000, seed: int = 0, y_max: float | None = None):
        self.n_max, self.seed, self.y_max = n_max, seed, y_max
        self.scaler = StandardScaler()
        kernel = ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(0.1)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)

    def fit(self, model: HybridCNNLSTM, x: np.ndarray, y: np.ndarray) -> "GPHead":
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(x), size=min(self.n_max, len(x)), replace=False)
        z = self.scaler.fit_transform(embed(model, x[idx]))
        self.gp.fit(z, y[idx])
        return self

    def predict(self, model: HybridCNNLSTM, x: np.ndarray) -> Prediction:
        z = self.scaler.transform(embed(model, x))
        mu, sd = self.gp.predict(z, return_std=True)
        if self.y_max is not None:
            mu = np.clip(mu, 0.0, self.y_max)
        return Prediction(mu, _floor(sd))


def gaussian_interval(p: Prediction, alpha: float):
    z = norm.ppf(1 - alpha / 2)
    return p.mu - z * p.sigma, p.mu + z * p.sigma


def conformal_quantile(y_cal: np.ndarray, p_cal: Prediction, alpha: float) -> float:
    """Normalized split-conformal quantile of |y - mu| / sigma."""
    scores = np.sort(np.abs(y_cal - p_cal.mu) / p_cal.sigma)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(scores[min(k, n) - 1])


def conformal_interval(p: Prediction, q: float):
    return p.mu - q * p.sigma, p.mu + q * p.sigma
