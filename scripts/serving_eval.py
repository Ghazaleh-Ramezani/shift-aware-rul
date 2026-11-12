"""Evaluate the served model end to end on the C-MAPSS test sets.

Builds the serving bundle on FD001 (or loads one with --model), then sends each test
subset through the same Predictor the container uses, as one request per subset, with
adaptation none, anchored and auto. Records what the drift monitor saw, which path
"auto" chose, and RMSE / conformal coverage at the last cycle of every engine.

    python scripts/serving_eval.py --config configs/quick.yaml --out results/serving
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rul import metrics
from rul.data import N_CONDITIONS, load_subset
from rul.experiment import load_config
from rul.serving import bundle as bundle_mod
from rul.serving.predictor import Predictor, to_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model", default=None, help="existing bundle directory (skips training)")
    ap.add_argument("--out", default="results/serving")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data_dir"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model) if args.model else out / "model"
    if not args.model:
        bundle_mod.save(bundle_mod.build(cfg, data_dir, seed=0), model_dir)
    pred = Predictor.from_dir(model_dir)

    rows = []
    for target in ["FD001", "FD002", "FD003", "FD004"]:
        test = load_subset(data_dir, target, cfg["cap"]).test
        truth = test.groupby("unit")["rul"].last()
        df = to_frame(test.drop(columns=["rul"]).to_dict("records"))
        for adaptation in ["none", "anchored", "auto"]:
            res = pred.predict(df, adaptation)
            y = truth.loc[[p["unit"] for p in res["predictions"]]].to_numpy()
            mu = np.array([p["rul"] for p in res["predictions"]])
            lo = np.array([p["lower"] for p in res["predictions"]])
            hi = np.array([p["upper"] for p in res["predictions"]])
            rows.append({"target": target, "requested": adaptation, "used": res["adaptation"],
                         "k_estimated": res["n_conditions"], "k_true": N_CONDITIONS[target],
                         "psi_settings": res["drift"]["max_psi_settings"],
                         "psi_sensors": res["drift"]["max_psi_sensors"], "drift": res["drift"]["level"],
                         "rmse": round(metrics.rmse(y, mu), 2), "coverage": round(metrics.picp(y, lo, hi), 3),
                         "engines": len(mu)})
            print(json.dumps(rows[-1]))

    (out / "results.json").write_text(json.dumps({"meta": pred.meta, "rows": rows}, indent=2, default=str))
    lines = [f"# Served model on the C-MAPSS test sets (source {pred.meta['source']}, seed {pred.meta['seed']}, "
             f"{pred.meta['ensemble_size']}-member ensemble, {1 - pred.meta['alpha']:.0%} conformal intervals)",
             "", "One request per test subset, through the same Predictor the container serves. "
             "`auto` switches to health-anchored normalization when the operating-settings PSI exceeds 0.25.", "",
             "| target | conditions (estimated / true) | PSI settings | PSI sensors | drift | requested | used | RMSE | coverage |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['target']} | {r['k_estimated']} / {r['k_true']} | {r['psi_settings']:.2f} | "
                     f"{r['psi_sensors']:.2f} | {r['drift']} | {r['requested']} | {r['used']} | {r['rmse']:.2f} | "
                     f"{r['coverage']:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
