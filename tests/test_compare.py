"""Checks of scripts/compare.py (D203) on tiny hand-made run folders: the best baseline,
the sign of the paired differences (accuracy and MAE), missing cells, and the stop when
two runs cannot be paired."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import compare


def fake_run(root, name, scores, y_train=(0, 1, 0, 1), task=None, metric="accuracy"):
    """A run folder with config.yaml, results.csv and y_*.npy per seed.

    scores: {(seed, probe, label_fraction): score}.
    """
    run = os.path.join(root, "runs", name)
    cfg = {"name": name, "dataset": "toy"} | ({"task": task} if task else {})
    rows = []
    for (seed, probe, frac), score in scores.items():
        os.makedirs(os.path.join(run, f"seed_{seed}"), exist_ok=True)
        np.save(os.path.join(run, f"seed_{seed}", "y_train.npy"), np.array(y_train))
        np.save(os.path.join(run, f"seed_{seed}", "y_test.npy"), np.array([1, 0]))
        other = "macro_f1" if metric == "accuracy" else "r2"
        rows.append({"seed": seed, "label_fraction": frac, "probe": probe, metric: score, other: 0.0})
    with open(os.path.join(run, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    pd.DataFrame(rows).to_csv(os.path.join(run, "results.csv"), index=False)
    return f"runs/{name}"


def write_config(root, settings):
    path = os.path.join(root, "compare.yaml")
    with open(path, "w") as f:
        yaml.safe_dump({"name": "compare_toy", "settings": settings}, f, sort_keys=False)
    return path


def output(root):
    folder = next(os.path.join(root, "runs", d) for d in os.listdir(os.path.join(root, "runs")) if "compare_toy" in d)
    return {n: pd.read_csv(os.path.join(folder, f"{n}.csv")) for n in ("summary", "paired")}


def test_best_baseline_and_paired_difference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = fake_run(tmp_path, "base_a", {(0, "logreg", 1.0): 0.80, (1, "logreg", 1.0): 0.84})  # mean 0.82
    b = fake_run(tmp_path, "base_b", {(0, "logreg", 1.0): 0.90, (1, "logreg", 1.0): 0.70,  # mean 0.80
                                      (0, "floor", 1.0): 0.5, (1, "floor", 1.0): 0.5})
    m = fake_run(tmp_path, "ssl", {(0, "logreg", 1.0): 0.85, (1, "logreg", 1.0): 0.80})
    compare.main(write_config(tmp_path, {"Toy": {"A": {"run": a, "role": "baseline"},
                                                 "B": {"run": b, "role": "baseline"},
                                                 "M": {"run": m, "role": "method"}}}))
    out = output(tmp_path)
    p = out["paired"][out["paired"].metric == "accuracy"].set_index("seed")
    assert set(p.best_baseline) == {"A"}  # highest mean, not highest single seed (B, seed 0)
    assert np.allclose(p.difference.loc[[0, 1]], [0.05, -0.04])  # per seed, M minus A
    assert "floor" not in set(out["paired"].probe)  # ignores the features: not paired (D83)
    s = out["summary"].set_index(["method", "probe"])
    assert s.loc[("A", "floor"), "n_seeds"] == 0  # not run by A: listed as missing (D13)
    assert s.loc[("B", "floor"), "n_seeds"] == 2


def test_mae_lower_is_better(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kw = {"task": "regression", "metric": "mae", "y_train": (0.1, 0.2, 0.3, 0.4)}
    a = fake_run(tmp_path, "base_a", {(0, "ridge", 1.0): 0.05}, **kw)
    b = fake_run(tmp_path, "base_b", {(0, "ridge", 1.0): 0.03}, **kw)
    m = fake_run(tmp_path, "ssl", {(0, "ridge", 1.0): 0.04}, **kw)
    compare.main(write_config(tmp_path, {"Toy": {"A": {"run": a, "role": "baseline"},
                                                 "B": {"run": b, "role": "baseline"},
                                                 "M": {"run": m, "role": "method"}}}))
    p = output(tmp_path)["paired"]
    row = p[p.metric == "mae"].iloc[0]
    assert row.best_baseline == "B"  # lowest MAE
    assert np.isclose(row.difference, -0.01)  # M is worse by 0.01: negative


def test_stops_when_labels_differ(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = fake_run(tmp_path, "base_a", {(0, "logreg", 1.0): 0.8})
    m = fake_run(tmp_path, "ssl", {(0, "logreg", 1.0): 0.8}, y_train=(1, 0, 0, 1))  # other order
    with pytest.raises(ValueError, match="differ in y_train"):
        compare.main(write_config(tmp_path, {"Toy": {"A": {"run": a, "role": "baseline"},
                                                     "M": {"run": m, "role": "method"}}}))
