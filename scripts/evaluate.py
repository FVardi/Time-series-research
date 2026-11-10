"""Stage 2: probes on stored representations (D26).

Usage:
    uv run python scripts/evaluate.py runs/<date_time>_<name>

Writes results.csv (one row per seed, label fraction and probe, with the probe's
fitting time, D79) and summary.csv (mean and standard deviation over seeds) into the
run folder, and prints the reproduction check against the published accuracy.
On datasets split by bearing (cv_groups) it also writes per_bearing.csv: the logreg
probe's test accuracy per test bearing, for every seed and label fraction (D160).
Configs with evaluation: per_class: true also get per_class.csv: the logreg probe's test
accuracy per class, for every seed and label fraction (D168, D179).
Configs with task: regression (kappa on Herning) use the regression probes and report
MAE and R² instead of accuracy and macro-F1 (D58, D59, D170).
XJTU-SY remaining useful life (D231, D232): evaluation: time_window: S embeds the representations of
S consecutive minutes of a bearing (DSCN Eq. 16); cv_groups with task: regression picks ridge's alpha
on cv_folds folds grouped by bearing; rul_metrics: true reports DSCN's score and RMSE (minutes) and
MAE and R² over the second half of each test bearing's life: their mean over the test bearings in
results.csv, the values per test bearing in per_bearing.csv.
"""

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from harness import evaluate, runs
from harness.data import canonical


def minutes_of(cfg, table):
    """Minute (file number) of each window and lifetime (minutes) of each run, from the captures
    table (XJTU-SY, D224: one window per one-minute capture, D232 item 1)."""
    caps = canonical.load_metadata(cfg["data_root"], cfg["dataset"])["captures"]
    minute = dict(zip(zip(caps.run_id, caps.capture_id), caps.file_number))
    lifetime = caps.groupby("run_id").size().to_dict()  # files per bearing = lifetime in minutes (D224, D228)
    return np.array([minute[k] for k in zip(table.run_id, table.capture_id)]), lifetime


def main(folder):
    cfg = runs.load_config(os.path.join(folder, "config.yaml"))
    ev = cfg["evaluation"]
    grouped = ev.get("cv_groups", False)  # D129; absent (FordA, FD configs) = folds as before
    regression = cfg.get("task") == "regression"  # D170
    per_class = ev.get("per_class", False)  # D168, D179; absent = no per_class.csv
    if grouped:  # group (bearing) of each training window, in the row order of Z_train (D33 window table)
        table = pd.read_csv(os.path.join(folder, "windows.csv"))
        groups_train = table[table.split == "train"].sort_values("row").group.to_numpy()
        groups_test = table[table.split == "test"].sort_values("row").group.to_numpy()  # for per_bearing.csv (D160)
    S = ev.get("time_window")  # D232 item 4; absent = each window on its own
    rul = ev.get("rul_metrics", False)  # D231 item 4, D232 item 6
    if S or rul:  # run and minute of every training and test row, in the row order of Z (D33 window table)
        table = pd.read_csv(os.path.join(folder, "windows.csv"))
        t_tr, t_te = (table[table.split == p].sort_values("row") for p in ("train", "test"))
        min_tr, lifetime = minutes_of(cfg, t_tr)
        min_te, _ = minutes_of(cfg, t_te)
        run_tr, run_te = t_tr.run_id.to_numpy(), t_te.run_id.to_numpy()
    if (grouped or per_class) and not regression:
        with open(os.path.join(folder, "classes.json")) as f:
            class_names = json.load(f)  # class k = class_names[k]
    rows = []
    bearing_rows = []  # per test bearing (D160), grouped datasets only
    class_rows = []  # per class (D168, D179), configs with per_class only
    for seed_dir in sorted(glob.glob(os.path.join(folder, "seed_*"))):
        seed = int(seed_dir.rsplit("_", 1)[1])
        Z_train, y_train = np.load(f"{seed_dir}/Z_train.npy"), np.load(f"{seed_dir}/y_train.npy")
        Z_test, y_test = np.load(f"{seed_dir}/Z_test.npy"), np.load(f"{seed_dir}/y_test.npy")
        g_tr, g_te = (groups_train, groups_test) if grouped else (None, None)
        r_tr, r_te, m_tr, m_te = (run_tr, run_te, min_tr, min_te) if (S or rul) else (None,) * 4
        if S:  # D232 item 4: [z_(n-S+1); ...; z_n]; minutes without S - 1 earlier ones are left out
            Z_train, k_tr = evaluate.time_window_embedding(Z_train, run_tr, min_tr, S)
            Z_test, k_te = evaluate.time_window_embedding(Z_test, run_te, min_te, S)
            y_train, y_test = y_train[k_tr], y_test[k_te]
            g_tr, g_te = (g_tr[k_tr], g_te[k_te]) if grouped else (None, None)
            r_tr, r_te, m_tr, m_te = run_tr[k_tr], run_te[k_te], min_tr[k_tr], min_te[k_te]
        for fraction in ev["label_fractions"]:
            idx = evaluate.label_subset(y_train, fraction, seed, stratify=not regression)
            for probe in ev["probes"]:
                start = time.perf_counter()
                draw, single = None, None
                if regression and probe == "ridge" and grouped:  # alpha on folds grouped by bearing (D232 item 5)
                    clf = evaluate.ridge_grouped(Z_train[idx], y_train[idx], g_tr[idx], seed, ev["cv_folds"])
                elif regression and probe == "ridge":  # alpha on shuffled folds, seeded (D170)
                    clf = evaluate.ridge(Z_train[idx], y_train[idx], seed)
                elif grouped and probe == "logreg":  # folds grouped by bearing, built on the labelled windows (D129)
                    clf, draw, single = evaluate.logreg_grouped(Z_train[idx], y_train[idx], g_tr[idx], seed)
                    single = ";".join(class_names[c] for c in single)  # classes with one labelled bearing (D148)
                else:
                    clf = evaluate.PROBES[probe](Z_train[idx], y_train[idx])
                fit_s = time.perf_counter() - start  # wall-clock time to fit the probe, incl. its cross-validation (D79)
                if rul:  # per test bearing, then the mean over the test bearings (D232 item 6)
                    pred = clf.predict(Z_test)
                    per = {b: evaluate.rul_scores(pred[r_te == b], y_test[r_te == b], m_te[r_te == b], lifetime[b])
                           for b in pd.unique(r_te)}
                    for b, r in per.items():
                        bearing_rows.append({"seed": seed, "label_fraction": fraction, "probe": probe, "bearing": b, **r})
                    score = {k: float(np.mean([r[k] for r in per.values()])) for k in ("score", "rmse_min", "mae", "r2")}
                    score["n_clipped"] = sum(r["n_clipped"] for r in per.values())
                elif regression:
                    score = evaluate.regression_scores(clf, Z_test, y_test)
                else:
                    score = evaluate.scores(clf, Z_test, y_test)
                rows.append({"seed": seed, "label_fraction": fraction, "n_labels": len(idx), "probe": probe,
                             **score, "probe_fit_s": fit_s,
                             "cv_draw": draw,  # grouped folds: the draw used (0 = first assignment valid)
                             "single_bearing_classes": single})
                print(rows[-1], flush=True)
                if grouped and probe == "logreg":  # per test bearing (D160)
                    for r in evaluate.per_bearing(clf, Z_test, y_test, g_te, class_names):
                        bearing_rows.append({"seed": seed, "label_fraction": fraction, "probe": probe, **r})
                if per_class and probe == "logreg":  # per class (D168, D179)
                    for r in evaluate.per_class(clf, Z_test, y_test, class_names):
                        class_rows.append({"seed": seed, "label_fraction": fraction, "probe": probe, **r})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(folder, "results.csv"), index=False)
    metrics = ["score", "rmse_min", "mae", "r2"] if rul else ["mae", "r2"] if regression else ["accuracy", "macro_f1"]
    summary = results.groupby(["probe", "label_fraction"])[metrics].agg(["mean", "std"])
    summary.to_csv(os.path.join(folder, "summary.csv"))
    print(summary)
    if bearing_rows:  # D160
        pd.DataFrame(bearing_rows).to_csv(os.path.join(folder, "per_bearing.csv"), index=False)
    if class_rows:  # D168, D179
        pd.DataFrame(class_rows).to_csv(os.path.join(folder, "per_class.csv"), index=False)

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
