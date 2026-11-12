"""Turn SageMaker data capture into drift metrics in CloudWatch.

Labels arrive only when engines fail, so interval coverage cannot be monitored in real
time; the input distribution can. This job reads the captured requests for a time
window, pools their cycles, scores them against the model's source baseline (PSI per
variable, as the endpoint does per request), and publishes:

    ShiftAwareRUL/MaxPSISettings          operating-settings shift (drives anchored adaptation)
    ShiftAwareRUL/MaxPSISensors           sensor shift (includes fault-mode and health-state shift)
    ShiftAwareRUL/AnchoredFraction        share of requests the endpoint routed to anchored normalization
    ShiftAwareRUL/CapturedRequests

An alarm on MaxPSISensors > 0.25 while MaxPSISettings stays low flags shift that the
anchored path does not address (e.g. a new fault mode, FD003 in this repository's study).

    python deploy/sagemaker/monitor.py --endpoint rul-shift-aware --capture s3://<bucket>/rul/capture \\
        --model model/ --since-hours 24
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rul.serving import bundle as bundle_mod  # noqa: E402
from rul.serving import drift  # noqa: E402
from rul.serving.predictor import to_frame  # noqa: E402

NAMESPACE = "ShiftAwareRUL"


def _decode(blob: dict[str, Any]) -> str:
    data = blob.get("data", "")
    return base64.b64decode(data).decode() if blob.get("encoding") == "BASE64" else data


def captured_pairs(s3, bucket: str, prefix: str, since: dt.datetime) -> list[tuple[str, str, str]]:
    """(content type, request body, response body) for every capture record written after ``since``."""
    pairs = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["LastModified"] < since or not obj["Key"].endswith(".jsonl"):
                continue
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode()
            for line in body.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)["captureData"]
                inp, out = rec["endpointInput"], rec.get("endpointOutput", {})
                pairs.append((inp.get("observedContentType", "application/json"), _decode(inp),
                              _decode(out) if out else ""))
    return pairs


def summarize(baseline: dict[str, Any], pairs: list[tuple[str, str, str]]) -> dict[str, Any]:
    frames, anchored, scored = [], 0, 0
    for ctype, body, output in pairs:
        try:
            if ctype.startswith("text/csv"):
                df = to_frame(pd.read_csv(io.StringIO(body)).to_dict("records"))
            else:
                df = to_frame(json.loads(body).get("records"))
        except Exception:
            continue                                  # rejected requests carry no usable cycles
        frames.append(df)
        if output:
            try:
                scored += 1
                anchored += json.loads(output).get("adaptation") == "anchored"
            except json.JSONDecodeError:
                pass
    if not frames:
        return {"requests": len(pairs), "usable": 0}
    report = drift.report(baseline, pd.concat(frames, ignore_index=True))
    return {"requests": len(pairs), "usable": len(frames), "rows": report["n_rows"],
            "max_psi_settings": report["max_psi_settings"], "max_psi_sensors": report["max_psi_sensors"],
            "level": report["level"], "anchored_fraction": round(anchored / scored, 4) if scored else 0.0,
            "psi": report["psi"]}


def publish(cw, endpoint: str, summary: dict[str, Any]) -> None:
    dims = [{"Name": "EndpointName", "Value": endpoint}]
    data = [{"MetricName": "CapturedRequests", "Value": float(summary["requests"]), "Unit": "Count", "Dimensions": dims}]
    if summary.get("usable"):
        data += [{"MetricName": name, "Value": float(summary[key]), "Unit": "None", "Dimensions": dims}
                 for name, key in (("MaxPSISettings", "max_psi_settings"), ("MaxPSISensors", "max_psi_sensors"),
                                   ("AnchoredFraction", "anchored_fraction"))]
    cw.put_metric_data(Namespace=NAMESPACE, MetricData=data)


def main(argv: list[str] | None = None) -> None:
    import boto3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--capture", required=True, help="the endpoint's DestinationS3Uri")
    ap.add_argument("--model", default="model", help="bundle directory (for the source baseline)")
    ap.add_argument("--since-hours", type=float, default=24)
    ap.add_argument("--dry-run", action="store_true", help="print the summary without publishing")
    args = ap.parse_args(argv)

    bucket, _, prefix = args.capture.removeprefix("s3://").partition("/")
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=args.since_hours)
    pairs = captured_pairs(boto3.client("s3"), bucket, f"{prefix.rstrip('/')}/{args.endpoint}/", since)
    summary = summarize(bundle_mod.load(args.model)["baseline"], pairs)
    print(json.dumps({k: v for k, v in summary.items() if k != "psi"}, indent=2))
    if not args.dry_run:
        publish(boto3.client("cloudwatch"), args.endpoint, summary)


if __name__ == "__main__":
    main()
