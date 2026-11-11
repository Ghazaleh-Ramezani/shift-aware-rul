"""SageMaker packaging, deployment and drift monitoring against moto (no AWS account needed)."""

import base64
import datetime as dt
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from rul.data import load_subset
from rul.serving import bundle as bundle_mod

ROOT = Path(__file__).resolve().parents[1]
REGION = "ca-central-1"
ROLE = "arn:aws:iam::123456789012:role/sagemaker-exec"
IMAGE = "123456789012.dkr.ecr.ca-central-1.amazonaws.com/shift-aware-rul:test"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "deploy" / "sagemaker" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


deploy_mod = _load("deploy")
monitor_mod = _load("monitor")


@pytest.fixture
def aws(monkeypatch):
    for k, v in {"AWS_DEFAULT_REGION": REGION, "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing"}.items():
        monkeypatch.setenv(k, v)
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket="rul-artifacts", CreateBucketConfiguration={"LocationConstraint": REGION})
        yield s3


@pytest.fixture(scope="module")
def model_dir(synthetic_dir, tmp_path_factory):
    from test_serving import TINY

    out = tmp_path_factory.mktemp("model")
    bundle_mod.save(bundle_mod.build(TINY, synthetic_dir, seed=0, log=lambda *_: None), out)
    return out


def test_package_puts_both_files_at_the_archive_root(aws, model_dir):
    deploy_mod.package(model_dir, "s3://rul-artifacts/rul/model.tar.gz", s3=aws)
    blob = aws.get_object(Bucket="rul-artifacts", Key="rul/model.tar.gz")["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == ["metadata.json", "model.pt"]


def test_deploy_creates_then_updates_the_endpoint(aws):
    sm = boto3.client("sagemaker", region_name=REGION)
    first = deploy_mod.deploy(sm, endpoint="rul-test", image=IMAGE, model_data="s3://rul-artifacts/rul/model.tar.gz",
                              role=ROLE, capture="s3://rul-artifacts/capture", tag="v1")
    assert first["action"] == "create" and first["data_capture"]
    cfg = sm.describe_endpoint_config(EndpointConfigName="rul-test-v1")
    assert cfg["DataCaptureConfig"]["DestinationS3Uri"] == "s3://rul-artifacts/capture"
    assert cfg["ProductionVariants"][0]["InstanceType"] == "ml.m5.large"
    model = sm.describe_model(ModelName="rul-test-v1")
    assert model["PrimaryContainer"]["Image"] == IMAGE

    # moto does not implement UpdateEndpoint; record the call instead.
    updates = []
    sm.update_endpoint = lambda **kw: updates.append(kw)
    second = deploy_mod.deploy(sm, endpoint="rul-test", image=IMAGE, model_data="s3://rul-artifacts/rul/model.tar.gz",
                               role=ROLE, tag="v2")
    assert second["action"] == "update" and not second["data_capture"]
    assert updates == [{"EndpointName": "rul-test", "EndpointConfigName": "rul-test-v2"}]
    assert not sm.describe_endpoint_config(EndpointConfigName="rul-test-v2").get("DataCaptureConfig")


def test_serverless_endpoints_reject_data_capture(aws):
    sm = boto3.client("sagemaker", region_name=REGION)
    with pytest.raises(ValueError, match="serverless"):
        deploy_mod.deploy(sm, endpoint="rul-sl", image=IMAGE, model_data="s3://rul-artifacts/m.tar.gz", role=ROLE,
                          capture="s3://rul-artifacts/capture", serverless_memory=2048)


def _capture_line(records, output):
    body = json.dumps({"records": records})
    return json.dumps({"captureData": {
        "endpointInput": {"observedContentType": "application/json", "mode": "INPUT",
                          "data": base64.b64encode(body.encode()).decode(), "encoding": "BASE64"},
        "endpointOutput": {"observedContentType": "application/json", "mode": "OUTPUT",
                           "data": json.dumps(output), "encoding": "JSON"}},
        "eventMetadata": {"eventId": "x"}})


def test_monitor_pools_captured_requests_and_publishes_drift(aws, model_dir, synthetic_dir):
    shifted = load_subset(synthetic_dir, "FD002").test.drop(columns=["rul"]).to_dict("records")
    in_domain = load_subset(synthetic_dir, "FD001").test.drop(columns=["rul"]).to_dict("records")
    lines = [_capture_line(shifted, {"adaptation": "anchored"}),
             _capture_line(in_domain, {"adaptation": "none"}),
             json.dumps({"captureData": {"endpointInput": {"observedContentType": "application/json",
                                                           "data": "{\"records\": []}", "encoding": "JSON"}}})]
    aws.put_object(Bucket="rul-artifacts", Key="capture/rul-test/AllTraffic/2026/11/11/10/a.jsonl",
                   Body="\n".join(lines).encode())
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
    pairs = monitor_mod.captured_pairs(aws, "rul-artifacts", "capture/rul-test/", since)
    summary = monitor_mod.summarize(bundle_mod.load(model_dir)["baseline"], pairs)
    assert summary["requests"] == 3 and summary["usable"] == 2
    assert summary["anchored_fraction"] == 0.5 and summary["level"] == "major"

    cw = boto3.client("cloudwatch", region_name=REGION)
    monitor_mod.publish(cw, "rul-test", summary)
    names = {m["MetricName"] for m in cw.list_metrics(Namespace=monitor_mod.NAMESPACE)["Metrics"]}
    assert names == {"CapturedRequests", "MaxPSISettings", "MaxPSISensors", "AnchoredFraction"}


def test_monitor_ignores_old_capture_files(aws, model_dir):
    aws.put_object(Bucket="rul-artifacts", Key="capture/rul-test/old.jsonl", Body=b"")
    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    assert monitor_mod.captured_pairs(aws, "rul-artifacts", "capture/rul-test/", future) == []
    assert monitor_mod.summarize(bundle_mod.load(model_dir)["baseline"], []) == {"requests": 0, "usable": 0}


def test_csv_capture_is_parsed(model_dir, synthetic_dir):
    csv = pd.DataFrame(load_subset(synthetic_dir, "FD001").test.drop(columns=["rul"])).to_csv(index=False)
    summary = monitor_mod.summarize(bundle_mod.load(model_dir)["baseline"], [("text/csv", csv, "")])
    assert summary["usable"] == 1 and summary["anchored_fraction"] == 0.0
