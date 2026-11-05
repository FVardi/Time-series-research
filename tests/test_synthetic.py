"""Tests for the synthetic data generator (harness/data/synthetic.py, D196)."""

import copy

import numpy as np
import pytest
import yaml

from harness import labels
from harness.data import canonical, synthetic


def config(name):
    return yaml.safe_load(open(f"configs/synthetic/{name}.yaml", encoding="utf-8"))


def small(cfg):
    """A small version for fast tests: fewer samples and levels, same model."""
    cfg = copy.deepcopy(cfg)
    cfg.update(n_samples=4096, speeds_rpm=[600, 2400], temperatures_c=[40, 80], n_bearings=2,
               noise_std_per_temperature=[0.2, 0.4])
    return cfg


@pytest.mark.parametrize("name", ["crossed", "confounded", "departures"])
def test_full_configs_give_144_runs(name):
    assert len(synthetic.run_table(config(name))) == 144


def test_crossed_has_every_combination():
    runs = synthetic.run_table(config("crossed"))
    combos = {(r["speed_rpm"], r["temperature_c"], r["state"], r["bearing"]) for r in runs}
    assert len(combos) == 4 * 3 * 3 * 4


def test_confounded_state_is_nominal_and_departures_differ():
    assert all(r["state"] == r["nominal_state"] for r in synthetic.run_table(config("confounded")))
    runs = synthetic.run_table(config("departures"))
    dep = [r for r in runs if r["departure"]]
    assert all(r["state"] != r["nominal_state"] for r in dep)
    assert all(r["state"] == r["nominal_state"] for r in runs if not r["departure"])
    assert 0.15 < len(dep) / len(runs) < 0.5  # departure_fraction 0.33


def test_nominal_state_follows_kappa_bins():
    for r in synthetic.run_table(config("confounded")):
        assert r["nominal_state"] == int(np.digitize(r["kappa_like"], [0.8, 1.5]))


def test_deterministic_and_seed_dependent():
    cfg = small(config("crossed"))
    run, nu = synthetic.run_table(cfg)[0], synthetic.bearing_nuisance(cfg)
    a, b = synthetic.generate_run(cfg, run, nu, 0), synthetic.generate_run(cfg, run, nu, 0)
    assert all(np.array_equal(a[s], b[s]) for s in synthetic.SENSORS)
    cfg2 = {**cfg, "seed": 1}
    c = synthetic.generate_run(cfg2, run, synthetic.bearing_nuisance(cfg2), 0)
    assert not np.array_equal(a["AE"], c["AE"])


def test_burst_rate_raises_ae_energy():
    """Sanity check of the state effect: with more and larger bursts, AE carries more energy (noise off)."""
    cfg = small(config("crossed"))
    cfg["sensors"]["AE"] = {**cfg["sensors"]["AE"], "periodic_weight": 0.0, "noise_weight": 0.0}
    cfg["states"] = {**cfg["states"], "burst_rate_hz": [10, 200, 400], "burst_amp": [0.3, 0.3, 0.3]}
    nu = synthetic.bearing_nuisance(cfg)
    runs = [r for r in synthetic.run_table(cfg) if r["bearing"] == "B0" and r["speed_rpm"] == 600
            and r["temperature_c"] == 40]
    energy = [np.mean(synthetic.generate_run(cfg, r, nu, i)["AE"].astype(float) ** 2) for i, r in enumerate(runs)]
    assert energy[0] < energy[1] < energy[2]


def test_speed_sets_shaft_line():
    """With bursts and noise off, the US spectrum peaks at the first shaft harmonic (amp / 1 is the largest line)."""
    cfg = small(config("crossed"))
    cfg["sensors"]["US"] = {**cfg["sensors"]["US"], "burst_weight": 0.0, "noise_weight": 0.0}
    cfg["bearing"] = {**cfg["bearing"], "resonance_mix": 0.0}
    nu = synthetic.bearing_nuisance(cfg)
    run = synthetic.run_table(cfg)[-1]  # 2400 rpm -> 40 Hz
    x = synthetic.generate_run(cfg, run, nu, 0)["US"].astype(float)
    f = np.fft.rfftfreq(len(x), 1 / cfg["sample_rate_hz"])
    assert abs(f[np.argmax(np.abs(np.fft.rfft(x)))] - run["speed_rpm"] / 60) < 3


def test_write_read_and_labels(tmp_path):
    cfg = small(config("departures"))
    runs, nu = synthetic.run_table(cfg), synthetic.bearing_nuisance(cfg)
    for i, run in enumerate(runs):
        tables, signals = synthetic.read(cfg, run, nu, i)
        canonical.write_run(str(tmp_path), cfg["dataset"], run["run_id"], tables, signals)
    meta = canonical.load_metadata(str(tmp_path), cfg["dataset"])
    caps = meta["captures"]
    assert len(caps) == len(runs) and set(meta["channels"].channel) == {"AE", "US"}
    x = canonical.load_captures(str(tmp_path), cfg["dataset"], caps.run_id[0], [caps.capture_id[0]], ["AE", "US"])[0]
    assert x.shape == (cfg["n_samples"], 2)
    y, g = labels.synthetic(caps)
    assert y == caps.state.tolist() and g == caps.bearing.tolist()
    k, _ = labels.synthetic_kappa(caps)
    assert np.allclose(k, caps.kappa_like)
