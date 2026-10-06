"""Merge runs that differ only in their seeds, then recompute summary and bootstrap.

    python scripts/merge_runs.py --runs results/s0 results/s1 results/s2 --out results/three_seeds
"""

import argparse

from rul.experiment import merge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merge(args.runs, args.out)
    print(f"merged {len(args.runs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
