"""Plot coverage (calibration) curves and RMSE from a finished run.

    python scripts/make_figures.py --results results/default
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, searchable SVGs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rul.metrics import LEVELS, coverage_curve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/default")
    ap.add_argument("--uq", default="ensemble", choices=["mcdropout", "ensemble", "gp"])
    args = ap.parse_args()
    res = Path(args.results)
    cfg = json.loads((res / "run_info.json").read_text())["config"]
    preds = np.load(res / "predictions.npz")
    df = pd.read_csv(res / "raw.csv")
    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)

    targets, adapts = cfg["targets"], cfg["adaptations"]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        for a in adapts:
            arr = np.concatenate([preds[f"{s}|{t}|{a}|{args.uq}"] for s in cfg["seeds"]], axis=1)
            ax.plot(LEVELS, coverage_curve(arr[0], arr[1], arr[2]), marker="o", ms=3, label=a)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{cfg['source']} → {t}")
        ax.set_xlabel("nominal coverage")
    axes[0].set_ylabel("observed coverage")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Calibration of {args.uq} intervals under shift", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(fig_dir / f"calibration_{args.uq}.{ext}", dpi=160)

    sub = df[df.uq == args.uq].groupby(["target", "adaptation"])["rmse"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(1.6 * len(targets) + 2, 3.2))
    width = 0.8 / len(adapts)
    for i, a in enumerate(adapts):
        s = sub[sub.adaptation == a].set_index("target").reindex(targets)
        ax.bar(np.arange(len(targets)) + i * width, s["mean"], width, yerr=s["std"].fillna(0), label=a, capsize=2)
    ax.set_xticks(np.arange(len(targets)) + width * (len(adapts) - 1) / 2, targets)
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title(f"Point accuracy by target ({args.uq})", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(fig_dir / f"rmse_{args.uq}.{ext}", dpi=160)
    cov = df[df.uq == args.uq].groupby(["target", "adaptation"])["picp_conformal"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(1.6 * len(targets) + 2, 3.2))
    for i, a in enumerate(adapts):
        s = cov[cov.adaptation == a].set_index("target").reindex(targets)
        ax.bar(np.arange(len(targets)) + i * width, s["mean"], width, yerr=s["std"].fillna(0), label=a, capsize=2)
    ax.axhline(1 - cfg["alpha"], color="k", ls="--", lw=0.8, label="nominal")
    ax.set_xticks(np.arange(len(targets)) + width * (len(adapts) - 1) / 2, targets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("observed coverage")
    ax.set_title(f"Conformal {1 - cfg['alpha']:.0%} intervals, calibrated on source only ({args.uq})", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(fig_dir / f"coverage_conformal_{args.uq}.{ext}", dpi=160)
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
