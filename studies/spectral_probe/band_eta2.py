"""Class contrast of each band (D270): how much of a band's relative energy is explained by the class?

For each set of encoder inputs in a spectral-probe config (studies/spectral_probe/configs/) with classes
(regression runs, e.g. XJTU-SY RUL, are skipped), the relative band energies of D244 (log share of the total
energy, per channel; the same targets as probe_spectral.py) are computed for the TRAINING windows, and for
every band and channel

    eta² = SS_between / SS_total

the standard effect size of a one-way ANOVA: SS_total = squared deviations of the windows from the grand mean,
SS_between = squared deviations of the class means from the grand mean, weighted by the class sizes.
0 = the band's relative energy has the same mean in every class; 1 = it is constant within each class.
Only the signal and the class labels are used; no representation and no probe.

One result per set of windows: FordA, CWRU, HUST one; Paderborn one per combination (its training bearings).

Usage (from the repo root):
    uv run --no-sync python studies/spectral_probe/band_eta2.py studies/spectral_probe/configs/cwru.yaml
Writes studies/spectral_probe/results/<config name>_eta2.csv, one row per (combination,) channel and band.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from harness import runs

sys.path.insert(0, os.path.dirname(__file__))
from probe_spectral import OUT, encoder_inputs, run_config, spectral_targets, window_set_key  # noqa: E402


def eta_squared(F, y):
    """eta² of every column of F (windows x targets) for the class labels y."""
    grand = F.mean(axis=0)
    ss_total = ((F - grand) ** 2).sum(axis=0)
    ss_between = sum((y == k).sum() * (F[y == k].mean(axis=0) - grand) ** 2 for k in np.unique(y))
    return ss_between / ss_total


def main(config_path):
    study = runs.load_config(config_path)
    rows, done, by_combination = [], set(), {}
    for folder in study["runs"]:
        folder = folder.rstrip("/")
        cfg = run_config(folder)
        key = window_set_key(folder)
        if key in done or cfg.get("task") == "regression":  # windows already scored, or no classes
            continue
        done.add(key)
        print(folder, flush=True)
        X, _, cfg = encoder_inputs(folder)
        rate = cfg.get("windows", {}).get("sample_rate_hz")
        F, info = spectral_targets(X["train"], study["n_bands"], ["relative"], cfg["channels"], rate)  # training windows only
        del X
        y = np.load(os.path.join(folder, f"seed_{study['seed']}", "y_train.npy"))  # the run's class labels (scripts/pretrain.py)
        eta2 = eta_squared(F, y)

        combination = None  # protocol runs (Paderborn, D220): the combination k of split.json
        split_path = os.path.join(folder, "split.json")
        if os.path.exists(split_path):
            with open(split_path) as f:
                combination = json.load(f)["combination"]
        # Runs of one combination whose windows differ only in scaling (e.g. another standardisation) give the
        # same relative energies; any other difference would make the join with the probe results ambiguous.
        if combination in by_combination:
            if not np.allclose(by_combination[combination], eta2):
                raise ValueError(f"{folder}: other training windows than an earlier run of combination {combination}")
            continue
        by_combination[combination] = eta2

        for j, t in enumerate(info):
            rows.append({"run": os.path.basename(folder), "combination": combination, **t,
                         "n_windows": len(y), "n_classes": len(np.unique(y)), "eta2": eta2[j]})
        best = int(np.argmax(eta2))
        print(f"  highest eta²: {info[best]['channel']} band {info[best]['band']}, {eta2[best]:.3f}", flush=True)

    path = os.path.join(OUT, f"{study['name']}_eta2.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Done: {path}")


if __name__ == "__main__":
    main(sys.argv[1])
