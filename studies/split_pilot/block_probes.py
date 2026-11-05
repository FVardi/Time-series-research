"""Block probes for the split-representation pilot (D184, D205): what does each part of the
representation contain?

For every finished run (run folder with results.csv) and every seed, the stored representations
(Z_train.npy / Z_test.npy) are cut into blocks and each block is probed for several targets:
  blocks : z_op = the first z_op_dims values, z_state = the rest, full = all (ts2vec_split runs);
           full only for runs without a split (plain TS2Vec); with live representations (D207)
           also z_op_live, z_state_live, full_live
  targets: the run's own class (synthetic: state; CWRU: fault class), at the config's label fractions;
           the factors at 100 % of the labels: synthetic speed_rpm, temperature_c, bearing; CWRU load_hp,
           bearing (the window's group = faulty bearing)
  probes : logreg (harness/evaluate.py) and floor; folds as the run's config (cv_groups: grouped by bearing)
A target whose test classes do not all occur in training (e.g. speed when a speed is held out) is
written as 'skipped'. On splits with departure runs, the state probe's predictions are also compared
with each test window's nominal state: pred_nominal = share of test windows predicted as the nominal
state (in a departure window the true state is never the nominal one, so a shortcut gives ~1).
Nothing is retrained.

Usage (quotes keep PowerShell from expanding the pattern; the script expands it):
    uv run --no-sync python studies/split_pilot/block_probes.py "runs/*_pilot_*" --seeds 0
--seeds: probe only these seeds of each run (the pilot uses seed 0 only, D206); without it, all stored seeds.
Writes: studies/split_pilot/results/block_probes/<run folder name>.csv
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from harness import evaluate, runs, windows
from harness.data import canonical

OUT = os.path.join(os.path.dirname(__file__), "results", "block_probes")
FACTORS = {"synthetic": ["speed_rpm", "temperature_c", "bearing"], "cwru": ["load_hp", "bearing"]}


def window_columns(cfg, table, columns):
    """Capture columns (captures and telemetry tables) for every window of the table, in its row order."""
    caps = windows.select(canonical.load_metadata(cfg["data_root"], cfg["dataset"]), {})
    have = [c for c in columns if c in caps.columns]
    return table[["run_id", "capture_id"]].merge(caps[["run_id", "capture_id"] + have], on=["run_id", "capture_id"],
                                                 how="left", validate="many_to_one")


def fit(Z, y, groups, seed, probe, grouped):
    """The harness probes, with folds as in scripts/evaluate.py (grouped by bearing if the config says so)."""
    if probe == "logreg" and grouped:
        return evaluate.logreg_grouped(Z, y, groups, seed)[0]
    return evaluate.PROBES[probe](Z, y)


def probe_run(folder, seeds=None):
    cfg = runs.load_config(os.path.join(folder, "config.yaml"))
    ev = cfg["evaluation"]
    grouped = ev.get("cv_groups", False)
    family = "synthetic" if cfg["dataset"].startswith("synthetic") else "cwru"
    table = pd.read_csv(os.path.join(folder, "windows.csv"), dtype={"run_id": str})
    train = table[table.split == "train"].sort_values("row").reset_index(drop=True)
    test = table[table.split == "test"].sort_values("row").reset_index(drop=True)
    extra = ["departure", "nominal_state"] if family == "synthetic" else []
    cols = {part: window_columns(cfg, t, FACTORS[family] + extra) for part, t in (("train", train), ("test", test))}
    if family == "cwru":  # the window's group is the faulty bearing (harness/labels.py, cwru)
        for part, t in (("train", train), ("test", test)):
            cols[part]["bearing"] = t.group.to_numpy()
    with open(os.path.join(folder, "classes.json")) as f:
        class_names = json.load(f)  # class k of y = class_names[k]
    k = cfg[cfg["method"]].get("z_op_dims") if cfg["method"] == "ts2vec_split" else None
    departures = family == "synthetic" and bool(cols["test"]["departure"].any())

    rows = []
    for seed_dir in sorted(glob.glob(os.path.join(folder, "seed_*"))):
        seed = int(seed_dir.rsplit("_", 1)[1])
        if seeds is not None and seed not in seeds:
            continue
        sources = {"": (np.load(f"{seed_dir}/Z_train.npy"), np.load(f"{seed_dir}/Z_test.npy"))}
        if os.path.exists(f"{seed_dir}/Z_train_live.npy"):  # D207: live encoder, blocks marked '_live'
            sources["_live"] = (np.load(f"{seed_dir}/Z_train_live.npy"), np.load(f"{seed_dir}/Z_test_live.npy"))
        y_train, y_test = np.load(f"{seed_dir}/y_train.npy"), np.load(f"{seed_dir}/y_test.npy")
        cuts = {"full": slice(None)} if k is None else {"z_op": slice(0, k), "z_state": slice(k, None),
                                                        "full": slice(None)}
        blocks = {name + suffix: (sl, Ztr, Zte) for suffix, (Ztr, Zte) in sources.items() for name, sl in cuts.items()}
        # targets: (name, training labels, test labels, label fractions)
        targets = [("class", y_train, y_test, ev["label_fractions"])]
        targets += [(c, cols["train"][c].astype(str).to_numpy(), cols["test"][c].astype(str).to_numpy(), [1.0])
                    for c in FACTORS[family]]
        for block, (sl, Z_train, Z_test) in blocks.items():
            assert len(Z_train) == len(train) and len(Z_test) == len(test)
            for target, ytr, yte, fractions in targets:
                if not set(yte) <= set(ytr):  # test classes the probe never saw in training
                    rows.append({"seed": seed, "block": block, "target": target, "status": "skipped"})
                    continue
                for fraction in fractions:
                    idx = evaluate.label_subset(ytr, fraction, seed)  # the same subsets as scripts/evaluate.py
                    for probe in ev["probes"]:
                        clf = fit(Z_train[idx, sl], ytr[idx], train.group.to_numpy()[idx], seed, probe, grouped)
                        row = {"seed": seed, "block": block, "target": target, "status": "ok",
                               "label_fraction": fraction, "n_labels": len(idx), "probe": probe,
                               **evaluate.scores(clf, Z_test[:, sl], yte)}
                        if target == "class" and departures:  # shortcut check: predicted = nominal state?
                            pred = np.array([class_names[c] for c in clf.predict(Z_test[:, sl])])
                            dep = cols["test"]["departure"].to_numpy().astype(bool)
                            nominal = cols["test"]["nominal_state"].astype(int).astype(str).to_numpy()
                            row["pred_nominal"] = float(np.mean(pred[dep] == nominal[dep]))
                        rows.append(row)
                        print(os.path.basename(folder), row, flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, os.path.basename(os.path.normpath(folder)) + ".csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="run folders or patterns, e.g. 'runs/*_pilot_*'")
    parser.add_argument("--seeds", type=int, nargs="+", help="probe only these seeds (D206: the pilot uses 0)")
    args = parser.parse_args()
    folders = sorted(f for p in args.patterns for f in glob.glob(p)
                     if os.path.isdir(f) and os.path.exists(os.path.join(f, "results.csv")))  # finished runs only
    print(f"{len(folders)} runs")
    for folder in folders:
        probe_run(folder, args.seeds)


if __name__ == "__main__":
    main()
