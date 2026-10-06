"""Synthetic C-MAPSS-format data so the tests run without downloading anything."""

import numpy as np
import pytest

CONDITIONS_6 = np.array([[0, 0, 100], [10, 0.25, 100], [20, 0.7, 100],
                         [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], dtype=float)


def _engine(rng, unit, length, n_cond):
    cyc = np.arange(1, length + 1)
    cond = rng.integers(0, n_cond, size=length) if n_cond > 1 else np.zeros(length, int)
    ops = (CONDITIONS_6[cond] if n_cond > 1 else np.tile([0.0, 0.0, 100.0], (length, 1)))
    ops = ops + rng.normal(0, 1e-3, ops.shape)
    health = (cyc / length) ** 2
    base = 100 + 20 * cond[:, None] + np.arange(21)[None, :]
    sensors = base + 5 * health[:, None] * rng.uniform(0.5, 1.5, 21) + rng.normal(0, 0.3, (length, 21))
    return np.column_stack([np.full(length, unit), cyc, ops, sensors])


def write_subset(root, name, n_cond, n_units=24, seed=0):
    rng = np.random.default_rng(seed)
    train = [_engine(rng, u, rng.integers(40, 80), n_cond) for u in range(1, n_units + 1)]
    test, rul = [], []
    for u in range(1, n_units // 2 + 1):
        full = _engine(rng, u, rng.integers(40, 80), n_cond)
        cut = rng.integers(15, len(full) - 5)
        test.append(full[:cut])
        rul.append(len(full) - cut)
    np.savetxt(root / f"train_{name}.txt", np.vstack(train), fmt="%.4f")
    np.savetxt(root / f"test_{name}.txt", np.vstack(test), fmt="%.4f")
    np.savetxt(root / f"RUL_{name}.txt", np.array(rul), fmt="%d")


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("cmapss")
    write_subset(root, "FD001", 1, seed=1)
    write_subset(root, "FD002", 6, seed=2)
    return root
