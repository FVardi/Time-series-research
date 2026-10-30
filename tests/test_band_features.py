"""Checks of the per-window band features (D165) and the pooled baseline (D166)."""

import numpy as np

from harness import band_features as bf
from harness.methods import pooled
from harness.methods.mssp_features import extract_features

FS = 2.5e6


def test_window_features_match_mssp_code():
    # Kurtosis, Hjorth parameters and energy equal the MSSP code's values (kurtosis, mobility,
    # complexity, rms**2) window by window. For a constant window both give the same result,
    # complexity NaN (rounding leaves its SD just above the MSSP code's 1e-30 guard).
    rng = np.random.default_rng(0)
    W = rng.standard_normal((5, bf.WINDOW)) * [[1], [2], [0.5], [3], [1]] + [[0], [0], [1], [0], [0]]
    W = np.vstack([W, np.full((1, bf.WINDOW), 0.7)])
    got = bf.window_features(W)
    for i, w in enumerate(W):
        ref = extract_features(w, FS)
        if i < 5:  # log of the mean square; the constant window's log energy is checked below
            assert np.isclose(got[i, 0], np.log(ref["rms"] ** 2))
        assert np.allclose(got[i, 1:], [ref["mobility"], ref["complexity"], ref["kurtosis"]], equal_nan=True)
    assert np.isclose(got[5, 0], np.log(0.7 ** 2))


def test_shapes_names_and_dropped_remainder():
    X = np.random.default_rng(1).standard_normal((2, 3 * bf.WINDOW + 100, 2))  # 3 full windows + a remainder
    Z = bf.sweep_features(X, ["AE", "UL"], FS)
    assert Z.shape == (2, 3, 12 * 4)
    names = bf.feature_names(["AE", "UL"])
    assert len(names) == 48 and names[0] == "AE 50-100 kHz log_energy" and names[-1] == "UL 89.1-178 kHz kurtosis"


def test_a_tone_lands_in_its_band():
    # A 150 kHz tone in AE and a 30 kHz tone in UL: most energy in AE 100-200 and UL 22.4-44.7 kHz;
    # Hjorth mobility of a sine is its angular frequency per sample, its complexity about 1.
    t = np.arange(20 * bf.WINDOW) / FS
    X = np.stack([np.sin(2 * np.pi * 150e3 * t), np.sin(2 * np.pi * 30e3 * t)], axis=1)[None]
    Z = bf.sweep_features(X, ["AE", "UL"], FS)[0].mean(axis=0)  # mean over windows
    names = bf.feature_names(["AE", "UL"])
    energy = {n.rsplit(" ", 1)[0]: Z[i] for i, n in enumerate(names) if n.endswith("log_energy")}
    ae = {k: v for k, v in energy.items() if k.startswith("AE")}
    ul = {k: v for k, v in energy.items() if k.startswith("UL")}
    assert max(ae, key=ae.get) == "AE 100-200 kHz" and max(ul, key=ul.get) == "UL 22.4-44.7 kHz"
    i = names.index("AE 100-200 kHz mobility")
    assert np.isclose(Z[i], 2 * np.pi * 150e3 / FS, rtol=0.01)
    assert np.isclose(Z[names.index("AE 100-200 kHz complexity")], 1, atol=0.05)


def test_pooled_is_the_mean_over_windows():
    X = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert np.allclose(pooled.encode(None, X), X.mean(axis=1))
