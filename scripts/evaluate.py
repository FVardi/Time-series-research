"""Stage 2: probes on stored representations (D26).

Usage:
    uv run python scripts/evaluate.py runs/<date_time>_<name>

Writes results.csv (one row per seed, label fraction and probe, with the probe's
fitting time, D79) and summary.csv (mean and standard deviation over seeds) into the
run folder, and prints the reproduction check against the published accuracy.
On datasets split by bearing (cv_groups) it also writes per_bearing.csv: the logreg
probe's test accuracy per test bearing, for every seed and label fraction (D160).
Configs with task: regression (kappa on Herning) use the regression probes and report
MAE and R² instead of accuracy and macro-F1 (D58, D59, D170).
"""

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from harness import evaluate, runs


def main(folder):
    cfg = runs.load_config(os.path.join(folder, "config.yaml"))
    ev = cfg["evaluation"]
    grouped = ev.get("cv_groups", False)  # D129; absent (FordA, FD configs) = folds as before
    regression = cfg.get("task") == "regression"  # D170
    if grouped:  # group (bearing) of each training window, in the row order of Z_train (D33 window table)
        table = pd.read_csv(os.path.join(folder, "windows.csv"))
        groups_train = table[table.split == "train"].sort_values("row").group.to_numpy()
        groups_test = table[table.split == "test"].sort_values("row").group.to_numpy()  # for per_bearing.csv (D160)
        with open(os.path.join(folder, "classes.json")) as f:
            class_names = json.load(f)  # class k = class_names[k]
    rows = []
    bearing_rows = []  # per test bearing (D160), grouped datasets only
    for seed_dir in sorted(glob.glob(os.path.join(folder, "seed_*"))):
        seed = int(seed_dir.rsplit("_", 1)[1])
        Z_train, y_train = np.load(f"{seed_dir}/Z_train.npy"), np.load(f"{seed_dir}/y_train.npy")
        Z_test, y_test = np.load(f"{seed_dir}/Z_test.npy"), np.load(f"{seed_dir}/y_test.npy")
        for fraction in ev["label_fractions"]:
            idx = evaluate.label_subset(y_train, fraction, seed, stratify=not regression)
            for probe in ev["probes"]:
                start = time.perf_counter()
                draw, single = None, None
                if regression and probe == "ridge":  # alpha on shuffled folds, seeded (D170)
                    clf = evaluate.ridge(Z_train[idx], y_train[idx], seed)
                elif grouped and probe == "logreg":  # folds grouped by bearing, built on the labelled windows (D129)
                    clf, draw, single = evaluate.logreg_grouped(Z_train[idx], y_train[idx], groups_train[idx], seed)
                    single = ";".join(class_names[c] for c in single)  # classes with one labelled bearing (D148)
                else:
                    clf = evaluate.PROBES[probe](Z_train[idx], y_train[idx])
                fit_s = time.perf_counter() - start  # wall-clock time to fit the probe, incl. its cross-validation (D79)
                score = evaluate.regression_scores(clf, Z_test, y_test) if regression else evaluate.scores(clf, Z_test, y_test)
                rows.append({"seed": seed, "label_fraction": fraction, "n_labels": len(idx), "probe": probe,
                             **score, "probe_fit_s": fit_s,
                             "cv_draw": draw,  # grouped folds: the draw used (0 = first assignment valid)
                             "single_bearing_classes": single})
                print(rows[-1], flush=True)
                if grouped and probe == "logreg":  # per test bearing (D160)
                    for r in evaluate.per_bearing(clf, Z_test, y_test, groups_test, class_names):
                        bearing_rows.append({"seed": seed, "label_fraction": fraction, "probe": probe, **r})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(folder, "results.csv"), index=False)
    metrics = ["mae", "r2"] if regression else ["accuracy", "macro_f1"]
    summary = results.groupby(["probe", "label_fraction"])[metrics].agg(["mean", "std"])
    summary.to_csv(os.path.join(folder, "summary.csv"))
    print(summary)
    if bearing_rows:  # D160
        pd.DataFrame(bearing_rows).to_csv(os.path.join(folder, "per_bearing.csv"), index=False)

    # Reproduction check (D26 item 6): the method's own probe (config: reproduction_probe, e.g.
    # svm_ts2vec for TS2Vec, ridge_minirocket for MiniRocket) with all labels vs the published accuracy.
    if ev["published_accuracy"] is None:  # e.g. PatchTST, or the raw/FFT/MSSP baselines: nothing published
        print("Reproduction: no published accuracy to compare with")
        return
    acc = results[(results.probe == ev["reproduction_probe"]) & (results.label_fraction == 1.0)].accuracy
    mean, sd, pub = acc.mean(), acc.std(), ev["published_accuracy"]
    ok = abs(pub - mean) <= max(2 * sd, 0.01)
    print(f"Reproduction: ours {mean:.4f} ± {sd:.4f} (n={len(acc)}), published {pub}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
