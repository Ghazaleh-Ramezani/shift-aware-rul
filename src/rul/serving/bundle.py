"""Train, save and load the model bundle served by the inference container.

A bundle holds two source-trained ensembles, one per normalization the predictor can
apply, each with its split-conformal quantile from held-out *source* engines:

* ``none``     global z-scoring with source statistics (the deployed model without adaptation)
* ``anchored`` statistics from the first ``anchor_cycles`` of each engine (health-anchored)

plus the drift baseline (decile bins of every sensor and operating setting on the
source fleet) and the configuration needed to rebuild windows at inference time.

    python -m rul.serving.bundle --config configs/quick.yaml --out model/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import __version__
from ..data import N_CONDITIONS, load_subset, split_units
from ..experiment import _env, _fit_source, load_config
from ..models import HybridCNNLSTM
from . import drift

BUNDLE_FILE = "model.pt"
METADATA_FILE = "metadata.json"
FORMAT_VERSION = 1


def build(cfg: dict, data_dir: str | Path, seed: int = 0, log=print) -> dict[str, Any]:
    """Train both source ensembles on ``cfg['source']`` and return the bundle as a dict."""
    src = load_subset(data_dir, cfg["source"], cfg["cap"])
    if N_CONDITIONS[cfg["source"]] != 1:
        raise ValueError("the served bundle assumes a single-condition source subset (e.g. FD001)")
    tr_units, _ = split_units(src.train["unit"].to_numpy(), cfg["val_frac"], seed)
    paths = {}
    for name, anchor in (("none", None), ("anchored", cfg["anchor_cycles"])):
        mode = "global" if anchor is None else "condition"
        fitted = _fit_source(src, mode, 1, tr_units, cfg, seed, log, anchor=anchor)
        paths[name] = {
            "state_dicts": [m.state_dict() for m in fitted["models"]],
            "q": float(fitted["q"]["ensemble"]),
            "mu": fitted["norm"].mu.tolist(),
            "sd": fitted["norm"].sd.tolist(),
        }
    meta = {
        "format_version": FORMAT_VERSION,
        "package_version": __version__,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": cfg["source"],
        "seed": seed,
        "cap": cfg["cap"],
        "window": cfg["window"],
        "alpha": cfg["alpha"],
        "anchor_cycles": cfg["anchor_cycles"],
        "ensemble_size": cfg["ensemble_size"],
        "model_kwargs": cfg["train"]["model_kwargs"],
        "n_features": 14,
        "conformal_q": {name: p["q"] for name, p in paths.items()},
        # Plain strings only: the bundle is loaded with weights_only=True.
        "env": {k: (None if v is None else str(v)) for k, v in _env().items()},
    }
    return {"meta": meta, "paths": paths, "baseline": drift.fit_baseline(src.train)}


def save(bundle: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out / BUNDLE_FILE)
    (out / METADATA_FILE).write_text(json.dumps(bundle["meta"], indent=2))
    return out


def load(model_dir: str | Path) -> dict[str, Any]:
    """Load a bundle; ``weights_only`` keeps unpickling limited to tensors and plain data."""
    bundle = torch.load(Path(model_dir) / BUNDLE_FILE, map_location="cpu", weights_only=True)
    if bundle["meta"]["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format {bundle['meta']['format_version']}")
    return bundle


def models_for(bundle: dict[str, Any], path: str) -> list[HybridCNNLSTM]:
    meta = bundle["meta"]
    out = []
    for state in bundle["paths"][path]["state_dicts"]:
        m = HybridCNNLSTM(meta["n_features"], **meta["model_kwargs"])
        m.load_state_dict(state)
        out.append(m.eval())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--data-dir", default=None, help="overrides data_dir in the config")
    ap.add_argument("--out", default="model")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    bundle = build(cfg, args.data_dir or cfg["data_dir"], seed=args.seed)
    print(f"bundle written to {save(bundle, args.out)}: conformal q = {bundle['meta']['conformal_q']}")


if __name__ == "__main__":
    main()
