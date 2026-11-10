"""Premise tests (a) and (c) of the spectral probe (D245, D252): how well do the spectral targets classify?

For each set of encoder inputs in a spectral-probe config (studies/spectral_probe/configs/), the
spectral targets of D244 (the same targets as probe_spectral.py, per channel) are used as features for
the benchmark's class probe, exactly as scripts/evaluate.py probes a representation: logreg (folds
grouped by bearing where the run's config sets cv_groups, D129) and the floor, seed 0, scored by test
accuracy and macro-F1. Regression runs (XJTU-SY RUL) are skipped. One result per set of windows
(FordA, CWRU, HUST: one; Paderborn: one per combination; XJTU-SY: the health-stage split).
  (a) all the targets together, at the run's own label fractions (stratified subsets, D26)   -> <name>_classify.csv
  (c) each band on its own: one feature, the band's relative energy (log share of the total), per
      channel, with all training windows (D252)                                                -> <name>_bands.csv

Usage (from the repo root):
    uv run --no-sync python studies/spectral_probe/classify_spectral.py studies/spectral_probe/configs/forda.yaml
Writes studies/spectral_probe/results/<config name>_classify.csv and <config name>_bands.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from harness import evaluate, runs

sys.path.insert(0, os.path.dirname(__file__))
from probe_spectral import OUT, encoder_inputs, run_config, spectral_targets, window_set_key  # noqa: E402


def class_probe(F_train, y_train, F_test, y_test, groups, ev, idx, seed):
    """Rows for logreg and floor on the features F (training rows idx), as scripts/evaluate.py probes a representation."""
    rows = []
    for probe in ("logreg", "floor"):
        if probe == "logreg" and ev.get("cv_groups", False):  # folds grouped by bearing (D129)
            clf, _, _ = evaluate.logreg_grouped(F_train[idx], y_train[idx], groups[idx], seed)
        else:
            clf = evaluate.PROBES[probe](F_train[idx], y_train[idx])
        rows.append({"label_fraction": len(idx) / len(y_train), "n_labels": len(idx), "probe": probe,
                     **evaluate.scores(clf, F_test, y_test)})
    return rows


def main(config_path):
    study = runs.load_config(config_path)
    seed = study["seed"]
    rows_a, rows_c, done = [], [], set()
    for folder in study["runs"]:
        folder = folder.rstrip("/")
        cfg = run_config(folder)
        key = window_set_key(folder)
        if key in done or cfg.get("task") == "regression":  # targets already classified, or no classes
            continue
        done.add(key)
        print(folder, flush=True)
        X, groups, cfg = encoder_inputs(folder)
        rate = cfg.get("windows", {}).get("sample_rate_hz")
        (F_train, info), (F_test, _) = (spectral_targets(X[p], study["n_bands"], study["targets"], cfg["channels"], rate)
                                        for p in ("train", "test"))
        del X
        seed_dir = os.path.join(folder, f"seed_{seed}")  # the run's class labels, 0..K-1 (scripts/pretrain.py)
        y_train, y_test = np.load(os.path.join(seed_dir, "y_train.npy")), np.load(os.path.join(seed_dir, "y_test.npy"))
        combination = None
        if os.path.exists(os.path.join(folder, "split.json")):
            with open(os.path.join(folder, "split.json")) as f:
                combination = json.load(f)["combination"]
        about = {"run": os.path.basename(folder), "combination": combination}
        ev = cfg["evaluation"]

        # (a): all targets as features, at the run's own label fractions (as its benchmark result)
        for fraction in ev["label_fractions"]:
            idx = evaluate.label_subset(y_train, fraction, seed)  # stratified by class (D26)
            for r in class_probe(F_train, y_train, F_test, y_test, groups, ev, idx, seed):
                rows_a.append({**about, "features": "spectral_targets", "n_features": F_train.shape[1],
                               **r, "label_fraction": fraction})
            print(f"  (a) {fraction:.0%} of the labels: accuracy {rows_a[-2]['accuracy']:.3f}", flush=True)

        # (c): each relative band energy on its own, all training windows (D252)
        everything = np.arange(len(y_train))
        for j, t in enumerate(info):
            if t["target"] != "relative":
                continue
            for r in class_probe(F_train[:, [j]], y_train, F_test[:, [j]], y_test, groups, ev, everything, seed):
                rows_c.append({**about, **t, **r})
        best = max((r for r in rows_c if r["run"] == about["run"] and r["probe"] == "logreg"), key=lambda r: r["accuracy"])
        print(f"  (c) best single band: {best['channel']} band {best['band']}, accuracy {best['accuracy']:.3f}", flush=True)

    for rows, suffix in ((rows_a, "classify"), (rows_c, "bands")):
        path = os.path.join(OUT, f"{study['name']}_{suffix}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Done: {path}")


if __name__ == "__main__":
    main(sys.argv[1])
