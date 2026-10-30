"""Checks of the Herning kappa labels (D55), the random split (D169) and the regression probes (D170)."""

import numpy as np
import pandas as pd

from harness import evaluate, kappa, windows
from harness.labels import HERNING, herning


def telemetry_rows():
    """Captures with telemetry: one analysed sweep and one per exclusion rule of D55."""
    return pd.DataFrame([
        # cmd Hz,  temp C, rpm target, temp target
        (1500 / 59.5, 60.0, 1500, 60),   # analysed
        (100 / 59.5, 60.0, 100, 60),     # bearing not rotating (rpm target 100)
        (0.0, 60.0, 0, 60),              # VFD off: rpm 0, outside 50-3600
        (1500 / 59.5, 30.0, 1500, 25),   # cool-down tail (temperature target 25)
        (1500 / 59.5, 37.0, 1500, 40),   # below 38 C
        (3700 / 59.5, 60.0, 3700, 60),   # above 3600 rpm
    ], columns=["telem_vfd_cmd_hz", "telem_omron_pv_c", "telem_rpm_target", "telem_temp_target"])


def test_kappa_labels_and_exclusions():
    labels, groups = herning(telemetry_rows())
    expected = kappa.kappa(1500.0, 60.0, HERNING["d_pw_mm"], HERNING["nu_40_cst"], HERNING["nu_100_cst"])
    assert np.isclose(labels[0], expected)
    assert labels[1:] == [None] * 5
    assert groups == [None] * 6


def test_random_parts_fixed_disjoint_and_sized():
    caps = pd.DataFrame({"run_id": "r", "capture_id": [f"sweep_{i:04d}" for i in range(101)]})
    a, b = windows.random_parts(caps, 0.2, seed=0), windows.random_parts(caps, 0.2, seed=0)
    assert list(a["test"].capture_id) == list(b["test"].capture_id)  # one fixed draw
    assert len(a["test"]) == 20 and len(a["train"]) == 81 and len(a["pretrain_only"]) == 0
    assert not set(a["test"].capture_id) & set(a["train"].capture_id)
    assert list(a["train"].capture_id) == sorted(a["train"].capture_id)  # original order kept


def test_regression_probes_and_unstratified_subset():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((300, 5))
    y = Z @ [1.0, -2.0, 0.5, 0.0, 0.0] + 0.01 * rng.standard_normal(300)
    idx = evaluate.label_subset(y, 0.1, seed=0, stratify=False)
    assert len(idx) == 30
    s = evaluate.regression_scores(evaluate.ridge(Z[:200], y[:200], seed=0), Z[200:], y[200:])
    assert s["r2"] > 0.99 and s["mae"] < 0.05
    f = evaluate.regression_scores(evaluate.floor_mean(Z[:200], y[:200]), Z[200:], y[200:])
    assert f["r2"] < 0.05  # the mean predicts nothing
