"""Data of a transfer setting: pretrain on a source run, fine-tune and test on a target run.

Used by the TF-C scripts (D35-D41). The four sets are official splits of two runs
of the same dataset, e.g. FD-A (source) and FD-B (target).
"""

import numpy as np
import pandas as pd

from harness.data import canonical

# (name, config key of the run, official split)
SETS = [("source_train", "source_run", "train"),
        ("target_train", "target_run", "train"),
        ("target_val", "target_run", "val"),
        ("target_test", "target_run", "test")]


def load_set(cfg, meta, run_id, split):
    """Inputs (n, time, channels), labels and capture IDs of one official split, in the files' order."""
    caps = meta["captures"]
    caps = caps[(caps.run_id == run_id) & (caps.official_split == split)].sort_values("capture_id")
    X = canonical.load_captures(cfg["data_root"], cfg["dataset"], run_id, caps.capture_id, cfg["channels"])
    return X, caps.label.to_numpy(), caps.capture_id.to_numpy()


def load_sets(cfg):
    """All four sets, as {name: (X, labels, capture_ids)}."""
    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    return {name: load_set(cfg, meta, cfg[run_key], split) for name, run_key, split in SETS}


def window_table(cfg, sets):
    """D33 window table. Pretraining and fine-tuning inputs both count as 'train'."""
    parts = []
    for name, run_key, split in SETS:
        X, _, ids = sets[name]
        parts.append(pd.DataFrame({
            "split": split,
            "set": name,
            "row": np.arange(len(ids)),
            "run_id": cfg[run_key],
            "capture_id": ids,
            "channels": ";".join(cfg["channels"]),
            "start": 0,
            "stop": X.shape[1],
        }))
    return pd.concat(parts, ignore_index=True)
