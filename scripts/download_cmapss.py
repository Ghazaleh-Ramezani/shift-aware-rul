"""Fetch the NASA C-MAPSS turbofan files into data/CMAPSSData.

The dataset is published by the NASA Prognostics Center of Excellence
(Saxena et al., PHM 2008). Two options:

  1. Download CMAPSSData.zip yourself from the NASA PCoE data repository and run
         python scripts/download_cmapss.py --zip path/to/CMAPSSData.zip
  2. Fetch the plain-text files from a public GitHub mirror (default):
         python scripts/download_cmapss.py

Row counts are checked against the published files either way.
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

MIRROR = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
EXPECTED_ROWS = {
    "train_FD001.txt": 20631, "test_FD001.txt": 13096, "RUL_FD001.txt": 100,
    "train_FD002.txt": 53759, "test_FD002.txt": 33991, "RUL_FD002.txt": 259,
    "train_FD003.txt": 24720, "test_FD003.txt": 16596, "RUL_FD003.txt": 100,
    "train_FD004.txt": 61249, "test_FD004.txt": 41214, "RUL_FD004.txt": 248,
}


def count_rows(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/CMAPSSData")
    ap.add_argument("--zip", default=None, help="path to the official CMAPSSData.zip")
    ap.add_argument("--mirror", default=MIRROR)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            for member in z.namelist():
                name = Path(member).name
                if name in EXPECTED_ROWS:
                    (out / name).write_bytes(z.read(member))
    else:
        for name in EXPECTED_ROWS:
            target = out / name
            if not target.exists():
                print(f"fetching {name}")
                urllib.request.urlretrieve(f"{args.mirror}/{name}", target)

    bad = [n for n, rows in EXPECTED_ROWS.items() if not (out / n).exists() or count_rows(out / n) != rows]
    if bad:
        raise SystemExit(f"missing or unexpected row counts: {bad}")
    print(f"all {len(EXPECTED_ROWS)} files present and verified in {out}")


if __name__ == "__main__":
    main()
