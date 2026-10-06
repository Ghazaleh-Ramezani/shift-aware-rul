"""Point, interval, and calibration metrics for RUL prediction."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

LEVELS = np.round(np.linspace(0.1, 0.9, 9), 2)


def rmse(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(np.mean((mu - y) ** 2)))


def nasa_score(y: np.ndarray, mu: np.ndarray) -> float:
    """PHM08 scoring function: late predictions are penalized more than early ones."""
    d = mu - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def picp(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Prediction-interval coverage probability."""
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean prediction-interval width (RUL cycles)."""
    return float(np.mean(hi - lo))


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2))


def coverage_curve(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                   levels: np.ndarray = LEVELS) -> np.ndarray:
    """Observed coverage of central Gaussian intervals at each nominal level."""
    z = np.abs(y - mu) / sigma
    return np.array([np.mean(z <= norm.ppf(0.5 + lv / 2)) for lv in levels])


def miscalibration_area(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                        levels: np.ndarray = LEVELS) -> float:
    """Mean |observed - nominal| coverage; 0 is perfectly calibrated."""
    return float(np.mean(np.abs(coverage_curve(y, mu, sigma, levels) - levels)))
