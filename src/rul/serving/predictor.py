"""Inference: from raw C-MAPSS-style cycles to RUL with conformal intervals.

``adaptation`` per request:

* ``none``     source normalization (what a model deployed without adaptation does)
* ``anchored`` per-condition statistics from the first ``anchor_cycles`` of each engine in
               the request, pooled per condition (transductive over the engines sent, as in
               the paper; send the fleet, or at least the engines' early cycles)
* ``auto``     ``anchored`` when the drift report shows an operating-settings shift
               (PSI > 0.25), otherwise ``none``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data import OP_COLS, SENSORS, Normalizer, make_windows
from ..uq import conformal_interval, ensemble
from . import bundle as bundle_mod
from . import drift

REQUIRED = ["unit", "cycle"] + OP_COLS + SENSORS
ADAPTATIONS = {"none", "anchored", "auto"}
MAX_CONDITIONS = 10


class InputError(ValueError):
    """The request cannot be scored as sent (HTTP 400)."""


def estimate_conditions(df: pd.DataFrame, min_share: float = 0.005) -> int:
    """Number of distinct operating regimes: settings rounded to their natural resolution."""
    rounded = pd.DataFrame({"a": df["op1"].round(0), "b": df["op2"].round(2), "c": df["op3"].round(0)})
    share = rounded.value_counts(normalize=True)
    return int(min(MAX_CONDITIONS, max(1, (share >= min_share).sum())))


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        raise InputError("'records' must be a non-empty list of cycle rows")
    df = pd.DataFrame.from_records(records)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputError(f"missing columns: {missing}")
    df = df[REQUIRED].copy()
    try:
        df = df.astype({c: "float64" for c in OP_COLS + SENSORS} | {"unit": "int64", "cycle": "int64"})
    except (TypeError, ValueError) as exc:
        raise InputError(f"non-numeric value: {exc}") from exc
    if not np.isfinite(df[OP_COLS + SENSORS].to_numpy()).all():
        raise InputError("non-finite sensor or setting value")
    if df.duplicated(["unit", "cycle"]).any():
        raise InputError("duplicate (unit, cycle) rows")
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


class Predictor:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.meta = bundle["meta"]
        self.models = {name: bundle_mod.models_for(bundle, name) for name in bundle["paths"]}

    @classmethod
    def from_dir(cls, model_dir) -> Predictor:
        return cls(bundle_mod.load(model_dir))

    def _features(self, df: pd.DataFrame, path: str, n_conditions: int) -> np.ndarray:
        if path == "none":
            p = self.bundle["paths"]["none"]
            mu, sd = np.asarray(p["mu"])[0], np.asarray(p["sd"])[0]
            return ((df[SENSORS].to_numpy(np.float64) - mu) / sd).astype(np.float32)
        anchor = self.meta["anchor_cycles"]
        early = df[df["cycle"] <= anchor]
        if early.empty:
            raise InputError(f"anchored normalization needs cycles 1-{anchor} of the engines; none were sent")
        norm = Normalizer("condition", n_conditions, seed=0, anchor_cycles=anchor).fit(df)
        labels = norm._labels(early)
        counts = np.bincount(labels, minlength=n_conditions)
        if (counts < 2).any():
            raise InputError("anchored normalization needs early cycles under every operating condition; "
                             "send more engines or their first cycles, or use adaptation='none'")
        return norm.transform(df)

    def predict(self, df: pd.DataFrame, adaptation: str = "auto", n_conditions: int | None = None) -> dict[str, Any]:
        if adaptation not in ADAPTATIONS:
            raise InputError(f"adaptation must be one of {sorted(ADAPTATIONS)}")
        report = drift.report(self.bundle["baseline"], df)
        k = n_conditions or estimate_conditions(df)
        path = adaptation if adaptation != "auto" else ("anchored" if report["settings_shift"] else "none")
        feats = self._features(df, path, k)
        x, _, units = make_windows(feats, df["unit"].to_numpy(), None, self.meta["window"], last_only=True)
        p = ensemble(self.models[path], x, self.meta["cap"])
        lo, hi = conformal_interval(p, self.bundle["paths"][path]["q"])
        cap = self.meta["cap"]
        seen = df.groupby("unit")["cycle"].count()
        preds = [{"unit": int(u), "rul": round(float(m), 2), "lower": round(float(max(lo_, 0.0)), 2),
                  "upper": round(float(min(hi_, cap)), 2), "sigma": round(float(s), 3),
                  "cycles_seen": int(seen[u])}
                 for u, m, lo_, hi_, s in zip(units, p.mu, lo, hi, p.sigma)]
        return {"adaptation": path, "requested_adaptation": adaptation, "n_conditions": k,
                "interval_level": round(1 - self.meta["alpha"], 3), "drift": report,
                "model": {k_: self.meta[k_] for k_ in ("source", "created_at", "package_version", "ensemble_size")},
                "predictions": preds}
