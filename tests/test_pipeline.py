import numpy as np
import torch

from rul import metrics, uq
from rul.data import Normalizer, load_subset, make_windows
from rul.experiment import run
from rul.models import HybridCNNLSTM
from rul.stats import paired_bootstrap
from rul.tta import adabn


def test_rul_labels_are_capped_and_decreasing(synthetic_dir):
    s = load_subset(synthetic_dir, "FD001", cap=30)
    assert s.train["rul"].max() <= 30 and s.train["rul"].min() == 0
    last = s.test.groupby("unit").tail(1)["rul"].to_numpy()
    expected = np.minimum(np.loadtxt(synthetic_dir / "RUL_FD001.txt"), 30)
    np.testing.assert_allclose(last, expected)


def test_windows_pad_short_engines():
    feats = np.arange(10, dtype=np.float32).reshape(5, 2)
    units = np.ones(5)
    x, y, u = make_windows(feats, units, np.arange(5, dtype=np.float32), window=8, last_only=True)
    assert x.shape == (1, 8, 2) and y[0] == 4
    np.testing.assert_array_equal(x[0, :4], np.repeat(feats[:1], 4, axis=0))


def test_condition_normalizer_centres_each_condition(synthetic_dir):
    s = load_subset(synthetic_dir, "FD002")
    z = Normalizer("condition", 6).fit(s.train).transform(s.train)
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 0.05


def test_anchored_normalizer_uses_only_early_cycles(synthetic_dir):
    s = load_subset(synthetic_dir, "FD001")
    norm = Normalizer("condition", 1, anchor_cycles=10).fit(s.train)
    early = s.train[s.train["cycle"] <= 10]
    z = norm.transform(early)
    assert abs(z.mean()) < 1e-6
    assert norm.transform(s.train).mean() > 0.1  # late-life drift survives normalization


def test_metrics_basic():
    y = np.array([10.0, 20.0])
    assert metrics.rmse(y, y) == 0
    assert metrics.nasa_score(y, y + 5) > metrics.nasa_score(y, y - 5)  # late is worse
    assert metrics.picp(y, y - 1, y + 1) == 1.0


def test_conformal_reaches_nominal_coverage():
    rng = np.random.default_rng(0)
    mu, sigma = rng.normal(size=4000), rng.uniform(0.5, 2, 4000)
    y = mu + sigma * rng.normal(size=4000) * 1.7  # sigma is miscalibrated by 1.7x
    p = uq.Prediction(mu, sigma)
    q = uq.conformal_quantile(y[:2000], uq.Prediction(mu[:2000], sigma[:2000]), alpha=0.1)
    lo, hi = uq.conformal_interval(uq.Prediction(mu[2000:], sigma[2000:]), q)
    assert abs(metrics.picp(y[2000:], lo, hi) - 0.9) < 0.03
    lo_g, hi_g = uq.gaussian_interval(p, 0.1)
    assert metrics.picp(y, lo_g, hi_g) < 0.75  # the uncalibrated Gaussian interval undercovers


def test_adabn_moves_input_statistics_and_leaves_source_untouched():
    torch.manual_seed(0)
    model = HybridCNNLSTM(3).eval()
    before = model.input_bn.running_mean.clone()
    x = (np.random.default_rng(0).normal(5.0, 1.0, (64, 10, 3))).astype(np.float32)
    adapted = adabn(model, x)
    assert torch.allclose(model.input_bn.running_mean, before)
    assert torch.allclose(adapted.input_bn.running_mean, torch.full((3,), 5.0), atol=0.2)


def test_paired_bootstrap_identical_inputs():
    a = np.random.default_rng(0).random(50)
    r = paired_bootstrap(a, a.copy(), n_boot=200)
    assert r["diff"] == 0 and r["p"] == 1.0


def test_end_to_end_tiny_run(synthetic_dir, tmp_path):
    cfg = {"source": "FD001", "targets": ["FD001", "FD002"],
           "adaptations": ["none", "adabn", "condnorm", "anchored"], "anchor_cycles": 10, "cap": 40, "window": 10, "alpha": 0.1,
           "val_frac": 0.25, "seeds": [0], "ensemble_size": 2, "mc_passes": 4, "gp_n_max": 80,
           "train": {"epochs": 2, "batch_size": 64, "lr": 1e-3, "weight_decay": 0.0, "patience": 2,
                     "model_kwargs": {"conv_channels": 8, "lstm_hidden": 8, "fusion_dim": 16, "dropout": 0.2}}}
    df = run(cfg, tmp_path, synthetic_dir, log=lambda *_: None)
    assert len(df) == 1 * 2 * 4 * 3
    assert np.isfinite(df[["rmse", "picp_conformal", "miscal"]].to_numpy()).all()
    for f in ["raw.csv", "predictions.npz", "bootstrap.csv", "summary.md", "run_info.json"]:
        assert (tmp_path / f).exists()
