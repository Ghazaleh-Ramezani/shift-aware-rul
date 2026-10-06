"""Run the full experiment from a YAML config.

    python scripts/run_experiment.py --config configs/default.yaml
"""

import argparse

from rul.experiment import load_config, run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None, help="overrides data_dir in the config")
    ap.add_argument("--out-dir", default=None, help="overrides out_dir in the config")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="overrides seeds (e.g. one seed per job)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    run(cfg, args.out_dir or cfg["out_dir"], args.data_dir or cfg["data_dir"])


if __name__ == "__main__":
    main()
