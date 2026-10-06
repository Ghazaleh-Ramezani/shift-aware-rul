"""Train on a source subset; evaluate every (adaptation x UQ method) pair on each target."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from . import metrics, tta, uq
from .data import N_CONDITIONS, Normalizer, load_subset, make_windows, split_units
from .models import train_model
from .stats import paired_bootstrap, rmse_from_sq

UQ_METHODS = ("mcdropout", "ensemble", "gp")
METRIC_COLS = ["rmse", "score", "picp_gauss", "picp_conformal", "mpiw_conformal", "nll", "miscal"]


def _env() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "git_sha": sha}


def _fit_source(src, mode, k, tr_units, cfg, seed, log, anchor=None):
    cap, win = cfg["cap"], cfg["window"]
    norm = Normalizer(mode, k, seed, anchor).fit(src.train[src.train["unit"].isin(tr_units)])
    feats = norm.transform(src.train)
    units, rul = src.train["unit"].to_numpy(), src.train["rul"].to_numpy()
    m = np.isin(units, tr_units)
    xtr, ytr, _ = make_windows(feats[m], units[m], rul[m], win)
    xva, yva, _ = make_windows(feats[~m], units[~m], rul[~m], win)
    models = []
    for i in range(cfg["ensemble_size"]):
        log(f"[seed {seed}] training member {i + 1}/{cfg['ensemble_size']} "
            f"({mode} normalization{', anchored' if anchor else ''})")
        models.append(train_model(xtr, ytr, xva, yva, y_scale=cap, seed=seed * 100 + i,
                                  log=log, **cfg["train"]))
    gp = uq.GPHead(cfg["gp_n_max"], seed, y_max=cap).fit(models[0], xtr, ytr)
    cal = {"mcdropout": uq.mc_dropout(models[0], xva, cap, cfg["mc_passes"], seed),
           "ensemble": uq.ensemble(models, xva, cap),
           "gp": gp.predict(models[0], xva)}
    # Conformal quantiles come from source validation engines only: no target labels.
    q = {name: uq.conformal_quantile(yva, p, cfg["alpha"]) for name, p in cal.items()}
    return {"norm": norm, "models": models, "gp": gp, "q": q}


def _base(adapt: str) -> str:
    """Normalization family of an adaptation name, e.g. "anchored+adabn" -> "anchored"."""
    base = adapt.replace("+adabn", "")
    if base == "adabn":
        return "none"
    if base not in {"none", "condnorm", "anchored"}:
        raise ValueError(f"unknown adaptation {adapt!r}")
    return base


def _row(y, p, q, alpha):
    lo_g, hi_g = uq.gaussian_interval(p, alpha)
    lo_c, hi_c = uq.conformal_interval(p, q)
    return {"rmse": metrics.rmse(y, p.mu), "score": metrics.nasa_score(y, p.mu),
            "picp_gauss": metrics.picp(y, lo_g, hi_g),
            "picp_conformal": metrics.picp(y, lo_c, hi_c),
            "mpiw_conformal": metrics.mpiw(lo_c, hi_c),
            "nll": metrics.gaussian_nll(y, p.mu, p.sigma),
            "miscal": metrics.miscalibration_area(y, p.mu, p.sigma)}, (lo_c, hi_c)


def run(cfg: dict, out_dir: str | Path, data_dir: str | Path, log=print) -> pd.DataFrame:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap, win, alpha = cfg["cap"], cfg["window"], cfg["alpha"]
    src = load_subset(data_dir, cfg["source"], cap)
    targets = {t: load_subset(data_dir, t, cap) for t in cfg["targets"]}
    k_src = N_CONDITIONS[cfg["source"]]

    rows, preds = [], {}
    for seed in cfg["seeds"]:
        tr_units, _ = split_units(src.train["unit"].to_numpy(), cfg["val_frac"], seed)
        bases = {_base(a) for a in cfg["adaptations"]}
        fitted = {"none": _fit_source(src, "global", 1, tr_units, cfg, seed, log)}
        if "condnorm" in bases:  # identical to global normalization when the source has one condition
            fitted["condnorm"] = (fitted["none"] if k_src == 1 else
                                  _fit_source(src, "condition", k_src, tr_units, cfg, seed, log))
        if "anchored" in bases:
            fitted["anchored"] = _fit_source(src, "condition", k_src, tr_units, cfg, seed, log,
                                             anchor=cfg["anchor_cycles"])

        for tname, tgt in targets.items():
            units_t, rul_t = tgt.test["unit"].to_numpy(), tgt.test["rul"].to_numpy()
            for adapt in cfg["adaptations"]:
                base = _base(adapt)
                art = fitted[base]
                if base == "none":
                    feats = art["norm"].transform(tgt.test)  # source statistics, no adaptation
                else:
                    # Per-condition statistics from unlabeled target sensors and settings;
                    # "anchored" uses only each engine's early (healthy) cycles.
                    anchor = cfg["anchor_cycles"] if base == "anchored" else None
                    feats = Normalizer("condition", N_CONDITIONS[tname], seed, anchor).fit(tgt.test).transform(tgt.test)
                x_last, y_last, _ = make_windows(feats, units_t, rul_t, win, last_only=True)
                models = art["models"]
                if "adabn" in adapt:
                    x_all, _, _ = make_windows(feats, units_t, None, win)  # unlabeled
                    models = [tta.adabn(m, x_all) for m in models]
                outs = {"mcdropout": uq.mc_dropout(models[0], x_last, cap, cfg["mc_passes"], seed),
                        "ensemble": uq.ensemble(models, x_last, cap),
                        "gp": art["gp"].predict(models[0], x_last)}
                for name, p in outs.items():
                    res, (lo_c, hi_c) = _row(y_last, p, art["q"][name], alpha)
                    rows.append({"seed": seed, "target": tname, "adaptation": adapt, "uq": name, **res})
                    preds[f"{seed}|{tname}|{adapt}|{name}"] = np.stack([y_last, p.mu, p.sigma, lo_c, hi_c])
                log(f"[seed {seed}] {tname:5s} {adapt:15s} ensemble RMSE "
                    f"{rows[-2]['rmse']:6.2f}  conformal coverage {rows[-2]['picp_conformal']:.2f}")

    df = pd.DataFrame(rows)
    finalize(out, cfg, df, preds, runtime_s=time.time() - t0)
    log(f"done in {time.time() - t0:.0f}s -> {out}")
    return df


def finalize(out: Path, cfg: dict, df: pd.DataFrame, preds: dict, runtime_s: float | None = None) -> None:
    """Write raw results, predictions, bootstrap comparisons, summary, and run info."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw.csv", index=False)
    np.savez_compressed(out / "predictions.npz", **preds)
    boot = _bootstrap(preds, cfg)
    boot.to_csv(out / "bootstrap.csv", index=False)
    (out / "summary.md").write_text(_summary_md(df, boot, cfg))
    info = {"config": cfg, "env": _env()}
    if runtime_s is not None:
        info["runtime_s"] = round(runtime_s, 1)
    (out / "run_info.json").write_text(json.dumps(info, indent=2))


def merge(run_dirs: list[str | Path], out_dir: str | Path) -> pd.DataFrame:
    """Combine runs that differ only in their seeds (e.g. one seed per job)."""
    cfgs, dfs, preds = [], [], {}
    for d in map(Path, run_dirs):
        cfgs.append(json.loads((d / "run_info.json").read_text())["config"])
        dfs.append(pd.read_csv(d / "raw.csv"))
        with np.load(d / "predictions.npz") as z:
            preds.update({k: z[k] for k in z.files})
    strip = [{k: v for k, v in c.items() if k not in {"seeds", "out_dir"}} for c in cfgs]
    if any(c != strip[0] for c in strip):
        raise ValueError("runs differ in more than their seeds")
    cfg = {**cfgs[0], "seeds": sorted({s for c in cfgs for s in c["seeds"]}), "out_dir": str(out_dir)}
    df = pd.concat(dfs, ignore_index=True)
    finalize(Path(out_dir), cfg, df, preds)
    return df


def _bootstrap(preds: dict, cfg: dict) -> pd.DataFrame:
    """Each adaptation vs. no adaptation, per target and UQ method, resampling engines."""
    out = []
    if "none" not in cfg["adaptations"]:
        return pd.DataFrame(out)
    for t in cfg["targets"]:
        for name in UQ_METHODS:
            def per_engine(adapt):
                arr = np.stack([preds[f"{s}|{t}|{adapt}|{name}"] for s in cfg["seeds"]])  # (S, 5, N)
                y, mu, lo, hi = arr[:, 0], arr[:, 1], arr[:, 3], arr[:, 4]
                return ((mu - y) ** 2).mean(0), ((y >= lo) & (y <= hi)).mean(0)
            sq0, cov0 = per_engine("none")
            for adapt in cfg["adaptations"]:
                if adapt == "none":
                    continue
                sq, cov = per_engine(adapt)
                r = paired_bootstrap(sq, sq0, fn=rmse_from_sq, seed=0)
                c = paired_bootstrap(cov, cov0, fn=np.mean, seed=0)
                out.append({"target": t, "uq": name, "adaptation": adapt,
                            "d_rmse": r["diff"], "d_rmse_lo": r["lo"], "d_rmse_hi": r["hi"], "p_rmse": r["p"],
                            "d_coverage": c["diff"], "d_cov_lo": c["lo"], "d_cov_hi": c["hi"], "p_coverage": c["p"]})
    return pd.DataFrame(out)


def _fmt(mean: float, std: float, nd: int, multi: bool) -> str:
    return f"{mean:.{nd}f} ± {std:.{nd}f}" if multi else f"{mean:.{nd}f}"


def _summary_md(df: pd.DataFrame, boot: pd.DataFrame, cfg: dict) -> str:
    multi = len(cfg["seeds"]) > 1
    g = df.groupby(["target", "adaptation", "uq"], sort=False)[METRIC_COLS]
    mean, std = g.mean(), g.std().fillna(0.0)
    lines = [f"# Results: source {cfg['source']}, {len(cfg['seeds'])} seed(s), "
             f"{cfg['ensemble_size']}-member ensemble, nominal coverage {1 - cfg['alpha']:.0%}", ""]
    for t in cfg["targets"]:
        lines += [f"## Target {t}", "",
                  "| adaptation | UQ | RMSE | NASA score | coverage (Gaussian) | coverage (conformal) | width (conformal) | miscal. area |",
                  "|---|---|---|---|---|---|---|---|"]
        for (tt, a, u), m in mean.iterrows():
            if tt != t:
                continue
            s = std.loc[(tt, a, u)]
            lines.append(f"| {a} | {u} | {_fmt(m.rmse, s.rmse, 2, multi)} | {_fmt(m.score, s.score, 0, multi)} | "
                         f"{_fmt(m.picp_gauss, s.picp_gauss, 2, multi)} | {_fmt(m.picp_conformal, s.picp_conformal, 2, multi)} | "
                         f"{_fmt(m.mpiw_conformal, s.mpiw_conformal, 1, multi)} | {_fmt(m.miscal, s.miscal, 3, multi)} |")
        lines.append("")
    lines += ["## Paired bootstrap vs. no adaptation (engines resampled, 95% CI)", "",
              "| target | UQ | adaptation | ΔRMSE [95% CI] | p | Δcoverage [95% CI] | p |", "|---|---|---|---|---|---|---|"]
    for _, r in boot.iterrows():
        lines.append(f"| {r.target} | {r.uq} | {r.adaptation} | {r.d_rmse:+.2f} [{r.d_rmse_lo:+.2f}, {r.d_rmse_hi:+.2f}] | "
                     f"{r.p_rmse:.3f} | {r.d_coverage:+.2f} [{r.d_cov_lo:+.2f}, {r.d_cov_hi:+.2f}] | {r.p_coverage:.3f} |")
    return "\n".join(lines) + "\n"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
