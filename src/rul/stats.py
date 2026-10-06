"""Paired bootstrap over engines."""

from __future__ import annotations

from typing import Callable

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray,
                     fn: Callable[[np.ndarray], float] = np.mean,
                     n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap fn(a) - fn(b), resampling engines jointly.

    `a` and `b` hold one value per engine (e.g. squared error), aligned by engine.
    Returns the observed difference, a 95% percentile interval, and a two-sided p-value.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be aligned per engine")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = np.array([fn(a[i]) - fn(b[i]) for i in idx])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"diff": float(fn(a) - fn(b)), "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)), "p": float(min(1.0, p))}


def rmse_from_sq(sq: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sq)))
