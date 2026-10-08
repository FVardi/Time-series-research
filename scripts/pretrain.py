"""Stage 1: pretrain an encoder per seed and store its representations (D10).

Usage:
    uv run python scripts/pretrain.py configs/ts2vec_forda.yaml

Creates runs/<date_time>_<name>/ and, per seed, seed_<k>/ containing the model,
the representations of the training and test series, and the training loss.
Only training inputs are used for pretraining; test inputs never are.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

from harness import runs
from harness.data import canonical
from harness.methods import tloss, ts2vec

METHODS = {"ts2vec": ts2vec, "tloss": tloss}


def load_split(cfg, meta, split):
    """Inputs, class labels and capture IDs of one official split."""
    caps = meta["captures"][meta["captures"].official_split == split]
    X = canonical.load_captures(cfg["data_root"], cfg["dataset"], cfg["run_id"], caps.capture_id, cfg["channels"])
    return X, caps.label.to_numpy(), caps.capture_id.to_numpy()


def window_table(cfg, split, capture_ids, n_samples):
    """Which data each representation was computed from (D33).

    Row i of Z_<split>.npy belongs to the row of this table with this split and row == i.
    Here every input is a whole capture, so start = 0 and stop = its length
    (stop is exclusive, as in canonical.load_signal).
    """
    return pd.DataFrame({
        "split": split,
        "row": np.arange(len(capture_ids)),
        "run_id": cfg["run_id"],
        "capture_id": capture_ids,
        "channels": ";".join(cfg["channels"]),
        "start": 0,
        "stop": n_samples,
    })


def main(config_path):
    cfg = runs.load_config(config_path)
    folder = runs.new_run_dir(config_path, cfg)
    method = METHODS[cfg["method"]]

    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    X_train, labels_train, ids_train = load_split(cfg, meta, "train")
    X_test, labels_test, ids_test = load_split(cfg, meta, "test")

    # Record where every representation comes from, and stop if a test capture
    # would be used for pretraining (D33). The table is the same for all seeds.
    windows = pd.concat([window_table(cfg, "train", ids_train, X_train.shape[1]),
                         window_table(cfg, "test", ids_test, X_test.shape[1])], ignore_index=True)
    runs.check_no_overlap(windows)
    windows.to_csv(os.path.join(folder, "windows.csv"), index=False)
    # Class labels -> 0..K-1, ordered as in the training set (as TS2Vec does).
    classes = np.unique(labels_train)
    y_train, y_test = np.searchsorted(classes, labels_train), np.searchsorted(classes, labels_test)

    for seed in cfg["seeds"]:
        print(f"seed {seed}: pretraining on {len(X_train)} training series", flush=True)
        out = os.path.join(folder, f"seed_{seed}")
        os.makedirs(out)
        runs.set_seed(seed)
        model, loss_log = method.pretrain(X_train, cfg[cfg["method"]], cfg["device"])
        method.save(model, os.path.join(out, "model"))
        np.save(os.path.join(out, "Z_train.npy"), method.encode(model, X_train))
        np.save(os.path.join(out, "Z_test.npy"), method.encode(model, X_test))
        np.save(os.path.join(out, "y_train.npy"), y_train)
        np.save(os.path.join(out, "y_test.npy"), y_test)
        runs.save_json(os.path.join(out, "loss.json"), [float(x) for x in loss_log])
    print(f"Done: {folder}")
    # The next step, ready to copy (forward slashes work in PowerShell too).
    print(f"Next: uv run python scripts/evaluate.py {folder.replace(os.sep, '/')}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    main(sys.argv[1])
