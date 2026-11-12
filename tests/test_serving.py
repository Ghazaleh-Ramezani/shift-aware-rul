"""Serving: bundle round trip, drift detection, adaptation routing, and the SageMaker contract."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rul.data import load_subset
from rul.serving import bundle as bundle_mod
from rul.serving import drift
from rul.serving.app import create_app
from rul.serving.predictor import InputError, Predictor, estimate_conditions, to_frame

TINY = {
    "source": "FD001", "cap": 30, "window": 8, "alpha": 0.1, "val_frac": 0.25, "anchor_cycles": 10,
    "ensemble_size": 2, "mc_passes": 2, "gp_n_max": 50,
    "train": {"epochs": 2, "batch_size": 64, "lr": 1e-3, "weight_decay": 0.0, "patience": 2,
              "model_kwargs": {"conv_channels": 8, "lstm_hidden": 8, "fusion_dim": 8, "dropout": 0.1}},
}


@pytest.fixture(scope="module")
def model_dir(synthetic_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    bundle_mod.save(bundle_mod.build(TINY, synthetic_dir, seed=0, log=lambda *_: None), out)
    return out


@pytest.fixture(scope="module")
def predictor(model_dir):
    return Predictor.from_dir(model_dir)


def records(synthetic_dir, subset):
    df = load_subset(synthetic_dir, subset, cap=30).test
    return df.drop(columns=["rul"]).to_dict("records")


def test_bundle_round_trip_reproduces_predictions(model_dir, synthetic_dir):
    a = Predictor(bundle_mod.load(model_dir))
    b = Predictor.from_dir(model_dir)
    df = to_frame(records(synthetic_dir, "FD001"))
    ra, rb = a.predict(df, "none"), b.predict(df, "none")
    assert ra["predictions"] == rb["predictions"]
    meta = json.loads((model_dir / bundle_mod.METADATA_FILE).read_text())
    assert set(meta["conformal_q"]) == {"none", "anchored"} and meta["window"] == 8


def test_predictions_are_bounded_and_intervals_contain_the_mean(predictor, synthetic_dir):
    out = predictor.predict(to_frame(records(synthetic_dir, "FD001")), "none")
    assert out["interval_level"] == 0.9 and len(out["predictions"]) == 12
    for p in out["predictions"]:
        assert 0 <= p["lower"] <= p["rul"] <= p["upper"] <= 30


def test_drift_is_stable_in_domain_and_major_under_condition_shift(predictor, synthetic_dir):
    base = predictor.bundle["baseline"]
    same = drift.report(base, load_subset(synthetic_dir, "FD001").train)
    shifted = drift.report(base, load_subset(synthetic_dir, "FD002").test)
    assert same["level"] == "stable" and not same["settings_shift"]
    assert shifted["level"] == "major" and shifted["settings_shift"]
    assert shifted["max_psi_settings"] > 1.0


def test_auto_adaptation_switches_to_anchored_only_under_shift(predictor, synthetic_dir):
    in_domain = predictor.predict(to_frame(records(synthetic_dir, "FD001")), "auto")
    shifted = predictor.predict(to_frame(records(synthetic_dir, "FD002")), "auto")
    assert in_domain["adaptation"] == "none"
    assert shifted["adaptation"] == "anchored" and shifted["n_conditions"] == 6


def test_estimate_conditions_counts_operating_regimes(synthetic_dir):
    assert estimate_conditions(load_subset(synthetic_dir, "FD001").test) == 1
    assert estimate_conditions(load_subset(synthetic_dir, "FD002").test) == 6


def test_anchored_needs_early_cycles(predictor, synthetic_dir):
    df = to_frame(records(synthetic_dir, "FD002"))
    late = df[df["cycle"] > 12]
    with pytest.raises(InputError, match="needs cycles"):
        predictor.predict(late, "anchored", n_conditions=6)


@pytest.mark.parametrize("bad, message", [
    ([], "non-empty"),
    ([{"unit": 1, "cycle": 1}], "missing columns"),
])
def test_input_validation(bad, message):
    with pytest.raises(InputError, match=message):
        to_frame(bad)


def test_duplicate_and_non_finite_rows_are_rejected(synthetic_dir):
    rows = records(synthetic_dir, "FD001")[:3]
    with pytest.raises(InputError, match="duplicate"):
        to_frame(rows + rows[:1])
    rows[0] = {**rows[0], "s2": float("nan")}
    with pytest.raises(InputError, match="non-finite"):
        to_frame(rows)


# ---------------------------------------------------------------- SageMaker contract


def test_ping_reports_unavailable_without_a_model(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        r = c.get("/ping")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"


def test_ping_and_json_invocation(model_dir, synthetic_dir):
    with TestClient(create_app(str(model_dir))) as c:
        assert c.get("/ping").json()["status"] == "ok"
        assert c.get("/execution-parameters").json()["MaxPayloadInMB"] == 6
        r = c.post("/invocations", json={"records": records(synthetic_dir, "FD002"), "adaptation": "auto"})
    assert r.status_code == 200
    body = r.json()
    assert body["adaptation"] == "anchored" and body["drift"]["level"] == "major"
    assert len(body["predictions"]) == 12


def test_csv_invocation_with_custom_attributes(model_dir, synthetic_dir):
    csv = pd.DataFrame(records(synthetic_dir, "FD001")).to_csv(index=False)
    with TestClient(create_app(str(model_dir))) as c:
        r = c.post("/invocations", content=csv, headers={
            "content-type": "text/csv", "X-Amzn-SageMaker-Custom-Attributes": "adaptation=none"})
    assert r.status_code == 200 and r.json()["adaptation"] == "none"


@pytest.mark.parametrize("payload, ctype, status", [
    (b"{not json", "application/json", 400),
    (b"[1, 2]", "application/json", 400),
    (json.dumps({"records": [{"unit": 1}]}).encode(), "application/json", 400),
    (json.dumps({"records": [], "adaptation": "x"}).encode(), "application/json", 400),
    (b"<xml/>", "application/xml", 415),
])
def test_bad_requests(model_dir, payload, ctype, status):
    with TestClient(create_app(str(model_dir))) as c:
        r = c.post("/invocations", content=payload, headers={"content-type": ctype})
    assert r.status_code == status, r.text


def test_unknown_adaptation_is_a_client_error(model_dir, synthetic_dir):
    with TestClient(create_app(str(model_dir))) as c:
        r = c.post("/invocations", json={"records": records(synthetic_dir, "FD001"), "adaptation": "magic"})
    assert r.status_code == 400 and "adaptation" in r.json()["error"]


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    df = pd.DataFrame({c: x for c in ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]})
    base = drift.fit_baseline(df)
    assert drift.report(base, df)["max_psi_sensors"] < 1e-6
