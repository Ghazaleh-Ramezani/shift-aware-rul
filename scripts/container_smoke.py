"""End-to-end check of the inference container, used by CI.

    python scripts/container_smoke.py bundle --out model          # tiny bundle on synthetic data
    docker run -d -p 8080:8080 -v $PWD/model:/opt/ml/model:ro shift-aware-rul serve
    python scripts/container_smoke.py check --url http://localhost:8080

The checks follow the SageMaker contract: /ping is 200 once the model is loaded, and
/invocations scores JSON and CSV requests, routes a shifted fleet to anchored
normalization, and answers malformed input with 400.
"""

import argparse
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from conftest import write_subset  # noqa: E402

TINY = {
    "source": "FD001", "cap": 30, "window": 8, "alpha": 0.1, "val_frac": 0.25, "anchor_cycles": 10,
    "ensemble_size": 2, "mc_passes": 2, "gp_n_max": 50,
    "train": {"epochs": 2, "batch_size": 64, "lr": 1e-3, "weight_decay": 0.0, "patience": 2,
              "model_kwargs": {"conv_channels": 8, "lstm_hidden": 8, "fusion_dim": 8, "dropout": 0.1}},
}


def synthetic(root: Path) -> Path:
    write_subset(root, "FD001", 1, seed=1)
    write_subset(root, "FD002", 6, seed=2)
    return root


def request(url: str, body: bytes, ctype: str, headers: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body, method="POST", headers={"content-type": ctype, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def check(url: str) -> None:
    from rul.data import load_subset

    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{url}/ping", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(2)
    else:
        sys.exit("/ping never returned 200")

    with tempfile.TemporaryDirectory() as tmp:
        root = synthetic(Path(tmp))
        fd001 = load_subset(root, "FD001").test.drop(columns=["rul"])
        fd002 = load_subset(root, "FD002").test.drop(columns=["rul"])

    status, body = request(f"{url}/invocations", json.dumps({"records": fd002.to_dict("records")}).encode(),
                           "application/json")
    assert status == 200, body
    assert body["adaptation"] == "anchored" and body["drift"]["settings_shift"], body["drift"]
    assert len(body["predictions"]) == fd002["unit"].nunique()

    status, body = request(f"{url}/invocations", fd001.to_csv(index=False).encode(), "text/csv",
                           {"X-Amzn-SageMaker-Custom-Attributes": "adaptation=auto"})
    assert status == 200 and body["adaptation"] == "none", body

    status, body = request(f"{url}/invocations", b"{not json", "application/json")
    assert status == 400, body
    print("container smoke test passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--out", default="model")
    c = sub.add_parser("check")
    c.add_argument("--url", default="http://localhost:8080")
    args = ap.parse_args()
    if args.command == "bundle":
        from rul.serving import bundle as bundle_mod

        with tempfile.TemporaryDirectory() as tmp:
            data = synthetic(Path(tmp))
            out = bundle_mod.save(bundle_mod.build(TINY, data, seed=0, log=lambda *_: None), args.out)
        print(f"bundle written to {out}")
    else:
        check(args.url)


if __name__ == "__main__":
    main()
