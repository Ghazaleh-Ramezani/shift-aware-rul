"""How much does health-anchored normalization depend on the target fleet and on the anchor length?

    python scripts/anchor_protocols.py --config configs/three_seeds.yaml \
        --anchors 5 10 20 30 50 --out results/anchor_protocols

In the main experiment the target normalizer is fitted on the unlabeled target
test set as one batch (transductive): the operating-condition clusters come from
the settings of every target cycle, and the per-condition statistics are pooled
over the first `anchor` cycles of every target engine. Target RUL is never used.

For each anchor length, this script trains the source models once per seed and
evaluates them under three target protocols:

* transductive  -- as in the main experiment;
* anchor_only   -- clusters and statistics both from cycles <= anchor, so nothing
                   after an engine's early life enters the normalizer;
* cross_fitted  -- anchor_only, fitted on one half of the target engines and
                   applied to the other half (2-fold, seeded): every engine is
                   normalized with statistics from other engines only, as a new
                   engine joining a fleet would be.

Conformal quantiles come from source validation engines only. Protocols are
compared on the same trained models. Absolute numbers depend on library versions
(run_info.json records them), so compare within this run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rul import tta, uq
from rul.data import (
    N_CONDITIONS,
    Normalizer,
    load_subset,
    make_windows,
    split_units,
)
from rul.experiment import _env, _fit_source, _row

PROTOCOLS = ("transductive", "anchor_only", "cross_fitted")
ADAPTATIONS = ("anchored", "anchored+adabn")


def target_features(tgt_df: pd.DataFrame, k: int, seed: int, anchor: int, protocol: str) -> np.ndarray:
    if protocol == "transductive":
        return Normalizer("condition", k, seed, anchor).fit(tgt_df).transform(tgt_df)
    early = tgt_df[tgt_df["cycle"] <= anchor]
    if protocol == "anchor_only":
        return Normalizer("condition", k, seed, None).fit(early).transform(tgt_df)
    # cross_fitted: each half of the fleet is normalized with the other half's early cycles.
    units = np.sort(tgt_df["unit"].unique())
    fold_a = np.random.default_rng(seed).permutation(units)[: len(units) // 2]
    in_a = tgt_df["unit"].isin(fold_a).to_numpy()
    early_in_a = early["unit"].isin(fold_a).to_numpy()
    out: np.ndarray | None = None
    for fit_rows, apply_rows in ((early_in_a, ~in_a), (~early_in_a, in_a)):
        part = Normalizer("condition", k, seed, None).fit(early[fit_rows]).transform(tgt_df[apply_rows])
        if out is None:
            out = np.empty((len(tgt_df), part.shape[1]))
        out[apply_rows] = part
    return out


def fmt(df: pd.DataFrame, value: str) -> pd.DataFrame:
    g = df.groupby(["key", "target"])[value].agg(["mean", "std"])
    return g.apply(lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}" if value == "rmse"
                   else f"{r['mean']:.2f}", axis=1).unstack("target")


def markdown(df: pd.DataFrame, main_anchor: int) -> str:
    lines = ["# Anchored normalization: target protocol and anchor length", "",
             ("Deep-ensemble RMSE (mean ± std over seeds) and 90% conformal coverage; "
              "source FD001; same trained models within each anchor length."), ""]
    main = df[df["anchor"] == main_anchor].copy()
    for adapt in ADAPTATIONS:
        sub = main[main["adaptation"] == adapt].copy()
        sub["key"] = sub["protocol"]
        r, c = fmt(sub, "rmse"), fmt(sub, "coverage")
        targets = list(r.columns)
        lines += [f"## Protocols at anchor = {main_anchor}: `{adapt}`", "",
                  "| Protocol | " + " | ".join(f"{t} RMSE | {t} cov." for t in targets) + " |",
                  "|---|" + "---|---|" * len(targets)]
        for p in PROTOCOLS:
            lines.append(f"| {p} | " + " | ".join(f"{r.loc[p, t]} | {c.loc[p, t]}" for t in targets) + " |")
        lines.append("")
    for adapt in ADAPTATIONS:
        sub = df[(df["protocol"] == "transductive") & (df["adaptation"] == adapt)].copy()
        sub["key"] = sub["anchor"]
        r, c = fmt(sub, "rmse"), fmt(sub, "coverage")
        targets = list(r.columns)
        lines += [f"## Anchor length (transductive): `{adapt}`", "",
                  "| Anchor cycles | " + " | ".join(f"{t} RMSE | {t} cov." for t in targets) + " |",
                  "|---|" + "---|---|" * len(targets)]
        for a in sorted(sub["anchor"].unique()):
            lines.append(f"| {a} | " + " | ".join(f"{r.loc[a, t]} | {c.loc[a, t]}" for t in targets) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/three_seeds.yaml")
    ap.add_argument("--data", default="data/CMAPSSData")
    ap.add_argument("--anchors", type=int, nargs="*", help="anchor lengths (default: the config's)")
    ap.add_argument("--out", default="results/anchor_protocols")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    anchors = args.anchors or [cfg["anchor_cycles"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap, win, alpha = cfg["cap"], cfg["window"], cfg["alpha"]
    src = load_subset(args.data, cfg["source"], cap)
    targets = {t: load_subset(args.data, t, cap) for t in cfg["targets"]}
    k_src = N_CONDITIONS[cfg["source"]]
    rows = []
    for anchor in anchors:
        for seed in cfg["seeds"]:
            tr_units, _ = split_units(src.train["unit"].to_numpy(), cfg["val_frac"], seed)
            art = _fit_source(src, "condition", k_src, tr_units, cfg, seed, lambda *a: None, anchor=anchor)
            for tname, tgt in targets.items():
                units_t, rul_t = tgt.test["unit"].to_numpy(), tgt.test["rul"].to_numpy()
                for protocol in PROTOCOLS:
                    feats = target_features(tgt.test, N_CONDITIONS[tname], seed, anchor, protocol)
                    x_last, y_last, _ = make_windows(feats, units_t, rul_t, win, last_only=True)
                    for adapt in ADAPTATIONS:
                        models = art["models"]
                        if adapt.endswith("adabn"):
                            x_all, _, _ = make_windows(feats, units_t, None, win)
                            models = [tta.adabn(m, x_all) for m in models]
                        p = uq.ensemble(models, x_last, cap)
                        res, _ = _row(y_last, p, art["q"]["ensemble"], alpha)
                        rows.append({"anchor": anchor, "seed": seed, "target": tname, "protocol": protocol,
                                     "adaptation": adapt, "rmse": res["rmse"],
                                     "coverage": res["picp_conformal"], "width": res["mpiw_conformal"]})
            print(f"anchor {anchor} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "raw.csv", index=False)
    main_anchor = cfg["anchor_cycles"] if cfg["anchor_cycles"] in anchors else anchors[0]
    text = markdown(df, main_anchor)
    (out / "summary.md").write_text(text + "\n")
    (out / "run_info.json").write_text(json.dumps(
        {"config": cfg, "anchors": anchors, "protocols": PROTOCOLS, "env": _env(),
         "runtime_s": round(time.time() - t0, 1)}, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
