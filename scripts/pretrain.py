"""Stage 1: pretrain an encoder per seed and store its representations (D10).

Usage:
    uv run python scripts/pretrain.py configs/ts2vec_forda.yaml

Creates runs/<date_time>_<name>/ and, per seed, seed_<k>/ containing the model,
the representations of the training and test series, the training loss and the
wall-clock time of each stage (timing.json, D79). The run folder also gets
hardware.json (where the times were measured).
Only training inputs are used for pretraining; test inputs never are.
A config with a 'split' section (D147, e.g. CWRU) is cut into windows by harness/windows.py:
pretraining then uses the training windows plus any 'pretrain_only' windows, and the run
stops if a test capture or test bearing would be used for pretraining (D121).
A config with 'rankme_epsilon' also stores RankMe of the training representations per seed
(rankme.json; D152), as a diagnostic only.
Baselines (raw, fft, envelope, catch22, minirocket, mssp) run through the same steps (D77).

Random-init baseline (D134): a config with 'random_init: as_initialised' or
'random_init: bn_from_train' stores the representations of the method's encoder as built,
without training (see harness/random_init.py). Without the key, the method is pretrained.
"""

import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

from harness import band_features, diagnostics, random_init, runs, windows
from harness.data import canonical
from harness.methods import (catch22, envelope, fft, frera, minirocket, mssp, patchtst, pooled, raw, simmtm, tfc_frozen,
                             tfjepa, tloss, ts2vec, tsjepa, tstcc)

METHODS = {"ts2vec": ts2vec, "tloss": tloss, "tsjepa": tsjepa, "patchtst": patchtst, "tstcc": tstcc,
           "tfjepa": tfjepa, "simmtm": simmtm, "frera": frera, "tfc": tfc_frozen,  # tfc: our TF-C (D122, D123)
           "raw": raw, "fft": fft, "catch22": catch22, "minirocket": minirocket,  # baselines (D75, D85)
           "envelope": envelope,  # squared envelope spectrum baseline (D161)
           "mssp": mssp,  # rig baseline (D49, D85)
           "pooled": pooled}  # feature sequence averaged over time (D166)
# Methods with an encoder to leave untrained (D134); their pretrain() takes train=False.
RANDOM_INIT_METHODS = ("ts2vec", "tloss", "tsjepa", "patchtst", "tstcc", "tfjepa", "simmtm", "frera", "tfc")


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


def windowed_inputs(cfg, meta, folder):
    """Inputs of a config with a 'split' section (D147), cut into windows (harness/windows.py).

    Returns the pretraining inputs ('train' and 'pretrain_only' windows, without labels, D126),
    the labelled training and test windows with their classes, and the window table (D33).
    Stops if a test capture or test bearing is in the pretraining data (D121). If the config
    says so, every window is standardised per channel with the mean and SD of the pretraining
    windows (D137); the statistics are saved in the run folder.
    """
    parts = windows.prepare(cfg, meta)
    (X_train, t_train), (X_test, t_test), (X_only, t_only) = parts["train"], parts["test"], parts["pretrain_only"]
    table = pd.concat([t_train, t_test, t_only], ignore_index=True)
    windows.check_separate(table)
    X_pre = np.concatenate([X_train, X_only])
    if cfg["windows"]["standardise"]:
        stats = windows.standardisation(X_pre)
        # the channels of X: the signal channels, or the 48 band features of D165
        names = band_features.feature_names(cfg["channels"]) if cfg["windows"].get("features") == "band" else cfg["channels"]
        runs.save_json(os.path.join(folder, "standardisation.json"),
                       {"channels": names, "mean": stats["mean"].tolist(), "std": stats["std"].tolist()})
        X_pre, X_train, X_test = (windows.standardise(X, stats) for X in (X_pre, X_train, X_test))
    print(f"windows: {len(X_train)} training, {len(X_test)} test, {len(X_only)} pretraining only "
          f"({len(X_pre)} for pretraining)", flush=True)
    return X_pre, X_train, t_train.label.to_numpy(), X_test, t_test.label.to_numpy(), table


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
    variant = cfg.get("random_init")  # D134; absent = pretrain as usual
    if variant is not None and (variant not in random_init.VARIANTS or cfg["method"] not in RANDOM_INIT_METHODS):
        raise ValueError(f"random_init must be one of {random_init.VARIANTS}, for one of {RANDOM_INIT_METHODS}")
    if hasattr(method, "check_settings"):  # stop before any work if a setting is still a placeholder
        method.check_settings(cfg[cfg["method"]])
    folder = runs.new_run_dir(config_path, cfg)
    runs.save_json(os.path.join(folder, "hardware.json"), hardware(cfg["device"]))
    if hasattr(method, "COMPILE_S"):  # code compiled when imported (MiniRocket, numba): recorded, not counted as fit time
        runs.save_json(os.path.join(folder, "compile.json"), {"compile_s": method.COMPILE_S})

    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    if "split" in cfg:  # split, windows and standardisation set in the config (D147)
        X_pre, X_train, labels_train, X_test, labels_test, table = windowed_inputs(cfg, meta, folder)
    else:  # whole captures with an official split (FordA, FD): unchanged
        X_train, labels_train, ids_train = load_split(cfg, meta, "train")
        X_test, labels_test, ids_test = load_split(cfg, meta, "test")
        X_pre = X_train  # pretraining on the training inputs only
        # Record where every representation comes from, and stop if a test capture
        # would be used for pretraining (D33). The table is the same for all seeds.
        table = pd.concat([window_table(cfg, "train", ids_train, X_train.shape[1]),
                           window_table(cfg, "test", ids_test, X_test.shape[1])], ignore_index=True)
        runs.check_no_overlap(table)
    table.to_csv(os.path.join(folder, "windows.csv"), index=False)
    if cfg.get("task") == "regression":  # a numeric target, e.g. kappa on Herning (D58): no classes
        y_train, y_test = np.asarray(labels_train, dtype=float), np.asarray(labels_test, dtype=float)
    else:
        # Class labels -> 0..K-1, ordered as in the training set (as TS2Vec does).
        classes = np.unique(labels_train)
        if not set(labels_test) <= set(classes):
            raise ValueError(f"test classes missing from training: {sorted(set(labels_test) - set(classes))}")
        runs.save_json(os.path.join(folder, "classes.json"), [str(c) for c in classes])  # class k = classes[k]
        y_train, y_test = np.searchsorted(classes, labels_train), np.searchsorted(classes, labels_test)

    for seed in cfg["seeds"]:
        print(f"seed {seed}: pretraining on {len(X_pre)} series", flush=True)
        out = os.path.join(folder, f"seed_{seed}")
        os.makedirs(out)
        runs.set_seed(seed)
        t0 = clock(cfg["device"])
        if variant is None:
            model, loss_log = method.pretrain(X_pre, cfg[cfg["method"]], cfg["device"])
        else:  # D134: the encoder as built, untrained; (b) also sets the batch-norm statistics from the pretraining inputs
            model, loss_log = method.pretrain(X_pre, cfg[cfg["method"]], cfg["device"], train=False)
            n_bn = random_init.recompute_batchnorm(model, method.encode, X_pre) if variant == "bn_from_train" else 0
            print(f"  random_init {variant}: untrained encoder, {n_bn} batch-norm layers set from the training series")
        t1 = clock(cfg["device"])
        Z_train = method.encode(model, X_train)
        t2 = clock(cfg["device"])
        Z_test = method.encode(model, X_test)
        t3 = clock(cfg["device"])
        method.save(model, os.path.join(out, "model"))
        np.save(os.path.join(out, "Z_train.npy"), Z_train)
        np.save(os.path.join(out, "Z_test.npy"), Z_test)
        if "rankme_epsilon" in cfg:  # D152: effective rank of the final training representations, a diagnostic only
            r = diagnostics.rankme(torch.from_numpy(np.asarray(Z_train, dtype=np.float64)), cfg["rankme_epsilon"])
            runs.save_json(os.path.join(out, "rankme.json"), {"rankme_Z_train": r, "n": len(Z_train),
                                                               "d": int(np.shape(Z_train)[1])})
            print(f"  RankMe of Z_train: {r:.1f} (of at most {min(np.shape(Z_train))})", flush=True)
        runs.save_json(os.path.join(out, "timing.json"), {
            "pretrain_s": t1 - t0,  # fitting or pretraining (random_init: building, plus the batch-norm pass in (b))
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
