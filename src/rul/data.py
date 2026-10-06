"""C-MAPSS loading, RUL labels, normalization, and sliding windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
# The 14 sensors that carry degradation information in C-MAPSS; the other
# seven are constant (or nearly so) within an operating condition.
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
           "s13", "s14", "s15", "s17", "s20", "s21"]
# Number of operating conditions in each subset.
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass
class Subset:
    name: str
    train: pd.DataFrame  # includes column "rul" (capped)
    test: pd.DataFrame   # includes column "rul" (true RUL, capped)


def read_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    return df


def _train_rul(df: pd.DataFrame, cap: float) -> np.ndarray:
    last = df.groupby("unit")["cycle"].transform("max")
    return np.minimum(last - df["cycle"], cap).to_numpy(dtype=np.float32)


def _test_rul(df: pd.DataFrame, rul_last: np.ndarray, cap: float) -> np.ndarray:
    units = np.sort(df["unit"].unique())
    if len(units) != len(rul_last):
        raise ValueError(f"{len(units)} test units but {len(rul_last)} RUL values")
    lookup = dict(zip(units, rul_last))
    last = df.groupby("unit")["cycle"].transform("max")
    rul = df["unit"].map(lookup) + (last - df["cycle"])
    return np.minimum(rul, cap).to_numpy(dtype=np.float32)


def load_subset(root: str | Path, name: str, cap: float = 125.0) -> Subset:
    """Load one C-MAPSS subset (e.g. "FD002") with capped RUL labels."""
    root = Path(root)
    train = read_txt(root / f"train_{name}.txt")
    test = read_txt(root / f"test_{name}.txt")
    rul_last = np.loadtxt(root / f"RUL_{name}.txt", ndmin=1)
    train["rul"] = _train_rul(train, cap)
    test["rul"] = _test_rul(test, rul_last, cap)
    return Subset(name, train, test)


class Normalizer:
    """Z-score the sensors, either globally or per operating condition.

    In "condition" mode the operating settings are clustered with k-means and
    each cluster gets its own mean and standard deviation. Fitting this on
    unlabeled target data is a label-free adaptation that uses metadata.

    With `anchor_cycles`, the statistics come only from each engine's first
    `anchor_cycles` cycles, when every engine is still healthy. This removes the
    operating-condition offset without also removing the degradation signal,
    which plain re-standardization does when the target population has a
    different mix of health states (e.g. truncated test trajectories).
    """

    def __init__(self, mode: str = "global", n_conditions: int = 1, seed: int = 0,
                 anchor_cycles: int | None = None):
        if mode not in {"global", "condition"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.k = n_conditions if mode == "condition" else 1
        self.seed = seed
        self.anchor_cycles = anchor_cycles
        self.km: KMeans | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def _labels(self, df: pd.DataFrame) -> np.ndarray:
        if self.k == 1:
            return np.zeros(len(df), dtype=int)
        assert self.km is not None
        return self.km.predict(df[OP_COLS].to_numpy())

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        if self.k > 1:
            self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(df[OP_COLS].to_numpy())
        if self.anchor_cycles is not None:
            df = df[df["cycle"] <= self.anchor_cycles]
        labels = self._labels(df)
        x = df[SENSORS].to_numpy(dtype=np.float64)
        self.mu = np.stack([x[labels == c].mean(0) for c in range(self.k)])
        sd = np.stack([x[labels == c].std(0) for c in range(self.k)])
        self.sd = np.where(sd < 1e-8, 1.0, sd)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.mu is not None and self.sd is not None, "call fit first"
        labels = self._labels(df)
        x = df[SENSORS].to_numpy(dtype=np.float64)
        return ((x - self.mu[labels]) / self.sd[labels]).astype(np.float32)


def make_windows(features: np.ndarray, units: np.ndarray, rul: np.ndarray | None,
                 window: int, last_only: bool = False):
    """Cut per-engine sliding windows of length `window`.

    Engines shorter than `window` are padded at the start with their first row.
    Returns X (N, window, F), y (N,) or None, and the unit id of each window.
    """
    xs, ys, us = [], [], []
    for unit in np.unique(units):
        idx = np.flatnonzero(units == unit)
        f = features[idx]
        r = None if rul is None else rul[idx]
        pad = max(0, window - len(f))
        if pad:
            f = np.concatenate([np.repeat(f[:1], pad, axis=0), f])
        w = sliding_window_view(f, (window, f.shape[1]))[:, 0]
        ends = np.arange(window - 1, len(f)) - pad  # index of the last real row
        if last_only:
            w, ends = w[-1:], ends[-1:]
        xs.append(w)
        if r is not None:
            ys.append(r[ends])
        us.append(np.full(len(w), unit))
    x = np.ascontiguousarray(np.concatenate(xs), dtype=np.float32)
    y = None if rul is None else np.concatenate(ys).astype(np.float32)
    return x, y, np.concatenate(us)


def split_units(units: np.ndarray, val_frac: float, seed: int):
    """Split engine ids into train and validation sets (no window leakage)."""
    rng = np.random.default_rng(seed)
    u = np.unique(units)
    rng.shuffle(u)
    n_val = max(1, int(round(val_frac * len(u))))
    return np.sort(u[n_val:]), np.sort(u[:n_val])
