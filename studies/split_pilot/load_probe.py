"""Step 0 of the split-representation pilot (D182): is the CWRU load recoverable from
existing frozen representations?

For each run given on the command line, and each seed, this fits the harness logreg probe
(harness.evaluate.logreg: standardise, L2 logistic regression, C by plain 5-fold CV) on the
training windows with load_hp (0-3 hp) as the target, and scores it on the test windows.
The floor (most frequent training load) is scored alongside. All labels are used (fraction 1.0).
Nothing is retrained: the probe reads the saved Z_train.npy / Z_test.npy of each seed.

Usage:  uv run python studies/split_pilot/load_probe.py runs/<run1> [runs/<run2> ...]
Writes: studies/split_pilot/results/load_probe.csv (one row per run, seed and probe).
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml

from harness import evaluate
from harness.data import canonical


def load_labels(run_dir):
    """load_hp of every training and test window, in the row order of Z_train / Z_test.

    windows.csv lists the windows per split in that order (run_id = CWRU file number);
    each CWRU file is recorded at one load, read from the canonical captures table.
    """
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml"), encoding="utf-8"))
    caps = canonical.load_metadata(cfg["data_root"], cfg["dataset"])["captures"]
    load_of_run = caps.drop_duplicates("run_id").set_index("run_id")["load_hp"]
    win = pd.read_csv(os.path.join(run_dir, "windows.csv"), dtype={"run_id": str})
    return {part: load_of_run.loc[win[win.split == part].run_id].to_numpy() for part in ("train", "test")}


def main(run_dirs):
    rows = []
    for run_dir in run_dirs:
        y = load_labels(run_dir)
        seeds = sorted(d for d in os.listdir(run_dir) if d.startswith("seed_"))
        for seed_dir in seeds:
            Z_train = np.load(os.path.join(run_dir, seed_dir, "Z_train.npy"))
            Z_test = np.load(os.path.join(run_dir, seed_dir, "Z_test.npy"))
            # guard: the windows table and the saved representations must line up
            assert len(Z_train) == len(y["train"]) and len(Z_test) == len(y["test"])
            for name in ("logreg", "floor"):
                clf = evaluate.PROBES[name](Z_train, y["train"])
                s = evaluate.scores(clf, Z_test, y["test"])
                rows.append({"run": os.path.basename(os.path.normpath(run_dir)), "seed": seed_dir,
                             "probe": name, **s})
                print(rows[-1])
    out = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "load_probe.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
