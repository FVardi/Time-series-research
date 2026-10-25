"""Stage 1: pretrain an encoder per seed and store its representations (D10).

Usage:
    uv run python scripts/pretrain.py configs/ts2vec_forda.yaml

Creates runs/<date_time>_<name>/ and, per seed, seed_<k>/ containing the model,
the representations of the training and test series, the training loss and the
wall-clock time of each stage (timing.json, D79). The run folder also gets
hardware.json (where the times were measured).
Only training inputs are used for pretraining; test inputs never are.
Baselines (raw, fft, catch22, minirocket, mssp) run through the same steps (D77).
"""

import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

from harness import runs
from harness.data import canonical
from harness.methods import (catch22, fft, frera, minirocket, mssp, patchtst, raw, simmtm, tfjepa, tloss, ts2vec,
                             tsjepa, tstcc)

METHODS = {"ts2vec": ts2vec, "tloss": tloss, "tsjepa": tsjepa, "patchtst": patchtst, "tstcc": tstcc,
           "tfjepa": tfjepa, "simmtm": simmtm, "frera": frera,
           "raw": raw, "fft": fft, "catch22": catch22, "minirocket": minirocket,  # baselines (D75, D85)
           "mssp": mssp}  # rig baseline (D49, D85)


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


def clock(device):
    """Wall-clock time in seconds (D79). On a GPU, waits for queued work first, so it is counted."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def hardware(device):
    """Where the times were measured (D79)."""
    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "device": device}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name()
    return info


def main(config_path):
    cfg = runs.load_config(config_path)
    method = METHODS[cfg["method"]]
    if hasattr(method, "check_settings"):  # stop before any work if a setting is still a placeholder
        method.check_settings(cfg[cfg["method"]])
    folder = runs.new_run_dir(config_path, cfg)
    runs.save_json(os.path.join(folder, "hardware.json"), hardware(cfg["device"]))
    if hasattr(method, "COMPILE_S"):  # code compiled when imported (MiniRocket, numba): recorded, not counted as fit time
        runs.save_json(os.path.join(folder, "compile.json"), {"compile_s": method.COMPILE_S})

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
        t0 = clock(cfg["device"])
        model, loss_log = method.pretrain(X_train, cfg[cfg["method"]], cfg["device"])
        t1 = clock(cfg["device"])
        Z_train = method.encode(model, X_train)
        t2 = clock(cfg["device"])
        Z_test = method.encode(model, X_test)
        t3 = clock(cfg["device"])
        method.save(model, os.path.join(out, "model"))
        np.save(os.path.join(out, "Z_train.npy"), Z_train)
        np.save(os.path.join(out, "Z_test.npy"), Z_test)
        runs.save_json(os.path.join(out, "timing.json"), {
            "pretrain_s": t1 - t0,  # fitting or pretraining
            "encode_train_s": t2 - t1,
            "encode_test_s": t3 - t2,
            "encode_series_per_s": (len(X_train) + len(X_test)) / (t3 - t1),
        })
        np.save(os.path.join(out, "y_train.npy"), y_train)
        np.save(os.path.join(out, "y_test.npy"), y_test)
        runs.save_json(os.path.join(out, "loss.json"), [float(x) for x in loss_log])
    print(f"Done: {folder}")
    # The next step, ready to copy (forward slashes work in PowerShell too).
    print(f"Next: uv run python scripts/evaluate.py {folder.replace(os.sep, '/')}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    main(sys.argv[1])
