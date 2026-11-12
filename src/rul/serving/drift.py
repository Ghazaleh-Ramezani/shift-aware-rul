"""Label-free shift detection: population stability index (PSI) against the source fleet.

After deployment the RUL labels arrive only when engines fail, so coverage cannot be
monitored directly; the findings in this repository show it collapses silently under
operating-condition shift. What *can* be monitored is the input distribution. The
baseline stores, for each sensor and for the three operating settings, decile bin edges
and bin proportions from the source training fleet; a request (or a day of captured
requests) is scored by PSI per variable.

Conventional reading of PSI: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
The predictor uses the max PSI over the operating settings to decide whether to switch
from source normalization to health-anchored normalization (``adaptation="auto"``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data import OP_COLS, SENSORS

N_BINS = 10
EPS = 1e-4
MODERATE, MAJOR = 0.1, 0.25


def _edges(x: np.ndarray) -> list[float]:
    qs = np.quantile(x, np.linspace(0, 1, N_BINS + 1)[1:-1])
    return np.unique(qs).tolist()


def _proportions(x: np.ndarray, edges: list[float]) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return np.clip(counts / max(len(x), 1), EPS, None)


def fit_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """Bin edges and proportions per variable, from the source training fleet."""
    out = {}
    for col in OP_COLS + SENSORS:
        x = df[col].to_numpy(dtype=np.float64)
        edges = _edges(x)
        out[col] = {"edges": edges, "p": _proportions(x, edges).tolist()}
    return {"n_rows": int(len(df)), "variables": out}


def psi(baseline_var: dict[str, Any], x: np.ndarray) -> float:
    p = np.asarray(baseline_var["p"])
    q = _proportions(x, baseline_var["edges"])
    return float(np.sum((q - p) * np.log(q / p)))


def report(baseline: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """PSI per variable; ``settings_shift`` drives the adaptation decision."""
    scores = {col: round(psi(spec, df[col].to_numpy(dtype=np.float64)), 4)
              for col, spec in baseline["variables"].items()}
    settings = max(scores[c] for c in OP_COLS)
    sensors = max(scores[c] for c in SENSORS)
    level = "major" if max(settings, sensors) > MAJOR else "moderate" if max(settings, sensors) > MODERATE else "stable"
    return {"psi": scores, "max_psi_settings": settings, "max_psi_sensors": sensors, "level": level,
            "settings_shift": settings > MAJOR, "n_rows": int(len(df))}
