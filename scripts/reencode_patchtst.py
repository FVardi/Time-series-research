"""Diagnostic (D86): PatchTST representations with another pooling, from the saved models of a finished run.

Question: does taking the last patch's vector (D74 item 7) limit PatchTST's probe results?
The pretrained models of a finished PatchTST run are loaded unchanged; only the step from the
patch vectors to one vector per series differs (pooling 'mean' or 'flatten', see
harness/methods/patchtst.py). Data, labels, seeds and probes are those of the source run.
Results are a diagnostic, not the PatchTST result (that stays the source run, D74).

Usage:
    uv run python scripts/reencode_patchtst.py runs/<source run> mean
    uv run python scripts/evaluate.py runs/<new run>

Creates runs/<date_time>_<source name>_<pooling>/ with config.yaml (the source run's config with
the new pooling and the source folder under 'diagnostic_of'), windows.csv (copied) and, per seed,
Z_train.npy, Z_test.npy, y_train.npy, y_test.npy (labels copied) and timing.json (encoding times).
The models themselves are not copied.
"""

import glob
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from harness import runs
from harness.data import canonical
from harness.methods import patchtst


def load_split(cfg, meta, split, windows):
    """Inputs of one official split, as scripts/pretrain.py loads them; stops if the series or their
    order differ from the source run's window table (so the copied labels belong to the same rows)."""
    caps = meta["captures"][meta["captures"].official_split == split]
    rows = windows[windows.split == split].sort_values("row")
    if list(caps.capture_id) != list(rows.capture_id):
        raise ValueError(f"{split} series differ from the source run's windows.csv")
    return canonical.load_captures(cfg["data_root"], cfg["dataset"], cfg["run_id"], caps.capture_id, cfg["channels"])


def clock(device):
    """Wall-clock time in seconds; on a GPU, waits for queued work first (as scripts/pretrain.py, D79)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def main(source, pooling):
    cfg = runs.load_config(os.path.join(source, "config.yaml"))
    if cfg["method"] != "patchtst":
        raise ValueError(f"{source} is not a PatchTST run")
    if pooling not in ("last", "mean", "flatten"):
        raise ValueError(f"pooling must be 'last', 'mean' or 'flatten', not {pooling!r}")
    s = dict(cfg["patchtst"], pooling=pooling)
    cfg = dict(cfg, name=f"{cfg['name']}_{pooling}", patchtst=s, diagnostic_of=source.replace(os.sep, "/"))

    folder = runs.new_run_dir(os.path.join(source, "config.yaml"), cfg)
    with open(os.path.join(folder, "config.yaml"), "w") as f:  # replace the copied config by the changed one
        f.write(f"# Diagnostic (D86): {cfg['diagnostic_of']} re-encoded with pooling '{pooling}'.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    shutil.copy(os.path.join(source, "windows.csv"), os.path.join(folder, "windows.csv"))

    windows = pd.read_csv(os.path.join(source, "windows.csv"))
    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    X_train = load_split(cfg, meta, "train", windows)
    X_test = load_split(cfg, meta, "test", windows)

    for seed_dir in sorted(glob.glob(os.path.join(source, "seed_*"))):
        out = os.path.join(folder, os.path.basename(seed_dir))
        os.makedirs(out)
        model = patchtst.load(os.path.join(seed_dir, "model"), X_train.shape[2], X_train.shape[1], s, cfg["device"])
        t1 = clock(cfg["device"])
        Z_train = patchtst.encode(model, X_train)
        t2 = clock(cfg["device"])
        Z_test = patchtst.encode(model, X_test)
        t3 = clock(cfg["device"])
        print(f"{os.path.basename(seed_dir)}: {Z_train.shape[1]} values per series", flush=True)
        np.save(os.path.join(out, "Z_train.npy"), Z_train)
        np.save(os.path.join(out, "Z_test.npy"), Z_test)
        for name in ("y_train.npy", "y_test.npy"):
            shutil.copy(os.path.join(seed_dir, name), os.path.join(out, name))
        runs.save_json(os.path.join(out, "timing.json"), {
            "encode_train_s": t2 - t1,
            "encode_test_s": t3 - t2,
            "encode_series_per_s": (len(X_train) + len(X_test)) / (t3 - t1),
        })
    print(f"Done: {folder}")
    print(f"Next: uv run python scripts/evaluate.py {folder.replace(os.sep, '/')}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    main(sys.argv[1].rstrip("/\\"), sys.argv[2])
