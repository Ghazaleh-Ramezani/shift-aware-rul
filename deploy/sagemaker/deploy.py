"""Package the model bundle and run it on a SageMaker real-time or serverless endpoint.

    # 1. model artifact: model.tar.gz (model.pt + metadata.json) uploaded to S3
    python deploy/sagemaker/deploy.py package --model model/ --s3 s3://<bucket>/rul/model.tar.gz

    # 2. endpoint (creates it, or rolls the existing one to a new endpoint config)
    python deploy/sagemaker/deploy.py deploy --endpoint rul-shift-aware \\
        --image <account>.dkr.ecr.<region>.amazonaws.com/shift-aware-rul:latest \\
        --model-data s3://<bucket>/rul/model.tar.gz --role arn:aws:iam::<account>:role/<sagemaker-exec> \\
        --instance-type ml.m5.large --capture s3://<bucket>/rul/capture

    # 3. a request
    python deploy/sagemaker/deploy.py invoke --endpoint rul-shift-aware --records request.json

Data capture (instance endpoints only) writes every request and response to S3, which
``deploy/sagemaker/monitor.py`` turns into drift metrics in CloudWatch.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

VARIANT = "AllTraffic"


def _split_s3(uri: str) -> tuple[str, str]:
    u = urlparse(uri)
    if u.scheme != "s3" or not u.netloc or not u.path.strip("/"):
        raise ValueError(f"not an s3://bucket/key URI: {uri}")
    return u.netloc, u.path.lstrip("/")


def package(model_dir: str | Path, s3_uri: str, s3=None) -> str:
    """Tar model.pt and metadata.json (at the archive root, as SageMaker expects) and upload."""
    import boto3

    model_dir = Path(model_dir)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("model.pt", "metadata.json"):
            tar.add(model_dir / name, arcname=name)
    bucket, key = _split_s3(s3_uri)
    (s3 or boto3.client("s3")).put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return s3_uri


def _endpoint_exists(sm, endpoint: str) -> bool:
    try:
        sm.describe_endpoint(EndpointName=endpoint)
        return True
    except sm.exceptions.ClientError as exc:
        if "Could not find" in str(exc) or "ValidationException" in str(exc):
            return False
        raise


def deploy(sm, *, endpoint: str, image: str, model_data: str, role: str, instance_type: str = "ml.m5.large",
           instance_count: int = 1, capture: str | None = None, capture_percent: int = 100,
           serverless_memory: int | None = None, tag: str | None = None) -> dict[str, Any]:
    """Create a model and endpoint config named after ``tag``; create or update the endpoint."""
    if serverless_memory and capture:
        raise ValueError("data capture is not available on serverless endpoints; use an instance endpoint")
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{endpoint}-{tag}"
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={"Image": image, "ModelDataUrl": model_data,
                          "Environment": {"SAGEMAKER_CONTAINER_LOG_LEVEL": "20"}},
    )
    variant: dict[str, Any] = {"VariantName": VARIANT, "ModelName": model_name}
    if serverless_memory:
        variant["ServerlessConfig"] = {"MemorySizeInMB": serverless_memory, "MaxConcurrency": 5}
    else:
        variant.update({"InstanceType": instance_type, "InitialInstanceCount": instance_count,
                        "InitialVariantWeight": 1.0})
    config: dict[str, Any] = {"EndpointConfigName": model_name, "ProductionVariants": [variant]}
    if capture:
        config["DataCaptureConfig"] = {
            "EnableCapture": True, "InitialSamplingPercentage": capture_percent,
            "DestinationS3Uri": capture,
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"], "CsvContentTypes": ["text/csv"]},
        }
    sm.create_endpoint_config(**config)
    action = "update" if _endpoint_exists(sm, endpoint) else "create"
    if action == "update":
        sm.update_endpoint(EndpointName=endpoint, EndpointConfigName=model_name)
    else:
        sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=model_name)
    return {"endpoint": endpoint, "model": model_name, "endpoint_config": model_name, "action": action,
            "data_capture": bool(capture), "serverless": bool(serverless_memory)}


def invoke(runtime, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = runtime.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                                   Accept="application/json", Body=json.dumps(payload).encode())
    return json.loads(resp["Body"].read())


def main(argv: list[str] | None = None) -> None:
    import boto3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("package")
    p.add_argument("--model", default="model")
    p.add_argument("--s3", required=True)
    d = sub.add_parser("deploy")
    d.add_argument("--endpoint", required=True)
    d.add_argument("--image", required=True)
    d.add_argument("--model-data", required=True)
    d.add_argument("--role", required=True)
    d.add_argument("--instance-type", default="ml.m5.large")
    d.add_argument("--capture", default=None, help="s3:// prefix for data capture (instance endpoints)")
    d.add_argument("--serverless-memory", type=int, default=None, help="MB; creates a serverless endpoint")
    d.add_argument("--wait", action="store_true")
    i = sub.add_parser("invoke")
    i.add_argument("--endpoint", required=True)
    i.add_argument("--records", required=True, help="JSON file with {'records': [...]} ")
    args = ap.parse_args(argv)

    if args.command == "package":
        print(package(args.model, args.s3))
    elif args.command == "deploy":
        sm = boto3.client("sagemaker")
        out = deploy(sm, endpoint=args.endpoint, image=args.image, model_data=args.model_data, role=args.role,
                     instance_type=args.instance_type, capture=args.capture,
                     serverless_memory=args.serverless_memory)
        print(json.dumps(out, indent=2))
        if args.wait:
            sm.get_waiter("endpoint_in_service").wait(EndpointName=args.endpoint)
            print("endpoint in service")
    else:
        payload = json.loads(Path(args.records).read_text())
        print(json.dumps(invoke(boto3.client("sagemaker-runtime"), args.endpoint, payload), indent=2))


if __name__ == "__main__":
    main()
