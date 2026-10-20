"""Stage 2: probes on stored representations (D26).

Usage:
    uv run python scripts/evaluate.py runs/<date_time>_<name>

Writes results.csv (one row per seed, label fraction and probe, with the probe's
fitting time, D79) and summary.csv (mean and standard deviation over seeds) into the
run folder, and prints the reproduction check against the published accuracy.
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from harness import evaluate, runs


def main(folder):
    cfg = runs.load_config(os.path.join(folder, "config.yaml"))
    ev = cfg["evaluation"]
    rows = []
    for seed_dir in sorted(glob.glob(os.path.join(folder, "seed_*"))):
        seed = int(seed_dir.rsplit("_", 1)[1])
        Z_train, y_train = np.load(f"{seed_dir}/Z_train.npy"), np.load(f"{seed_dir}/y_train.npy")
        Z_test, y_test = np.load(f"{seed_dir}/Z_test.npy"), np.load(f"{seed_dir}/y_test.npy")
        for fraction in ev["label_fractions"]:
            idx = evaluate.label_subset(y_train, fraction, seed)
            for probe in ev["probes"]:
                start = time.perf_counter()
                clf = evaluate.PROBES[probe](Z_train[idx], y_train[idx])
                fit_s = time.perf_counter() - start  # wall-clock time to fit the probe, incl. its cross-validation (D79)
                rows.append({"seed": seed, "label_fraction": fraction, "n_labels": len(idx), "probe": probe,
                             **evaluate.scores(clf, Z_test, y_test), "probe_fit_s": fit_s})
                print(rows[-1], flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(folder, "results.csv"), index=False)
    summary = results.groupby(["probe", "label_fraction"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(folder, "summary.csv"))
    print(summary)

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
