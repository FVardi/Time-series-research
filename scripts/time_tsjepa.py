"""Timing check: seconds per pretraining epoch for TS-JEPA (to plan the learning-rate search, D66).

Usage:
    uv run python scripts/time_tsjepa.py configs/tsjepa_forda.yaml configs/tsjepa_forda_fixed.yaml

For each config: one warm-up epoch (excluded; the first epoch includes start-up costs),
then n_epochs timed epochs on the config's data and device. Prints seconds per epoch and
the projected time for the config's full number of epochs. Writes no run folder.

The learning rate is still open in the configs (D66), so a fixed value is used here
(TIMING_LR). It only affects the weights, not the time per epoch; nothing from this
script is used as a result.
"""

import sys
import time

import torch

from harness import runs
from harness.data import canonical
from harness.methods import tsjepa

TIMING_LR = 1e-4  # for timing only (the learning rate does not change the computation per step)
N_EPOCHS = 3  # timed epochs per config


def epochs_time(X, s, device, n_epochs):
    """Seconds for pretraining with n_epochs epochs (waits for the GPU to finish)."""
    t = time.time()
    tsjepa.pretrain(X, dict(s, epochs=n_epochs), device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.time() - t


def main(config_paths):
    for path in config_paths:
        cfg = runs.load_config(path)
        s = cfg["tsjepa"]
        s["optimizer"] = dict(s["optimizer"], lr=TIMING_LR)
        s["print_every"] = 10**9  # only the first epoch's loss is printed (epoch 0)
        meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
        caps = meta["captures"][meta["captures"].official_split == "train"]
        X = canonical.load_captures(cfg["data_root"], cfg["dataset"], cfg["run_id"], caps.capture_id, cfg["channels"])
        runs.set_seed(0)
        epochs_time(X, s, cfg["device"], 1)  # warm-up
        per_epoch = epochs_time(X, s, cfg["device"], N_EPOCHS) / N_EPOCHS
        full = per_epoch * s["epochs"] / 3600
        print(f"{cfg['name']} (masking {s['masking']}, device {cfg['device']}): {per_epoch:.2f} s per epoch; "
              f"{s['epochs']} epochs = {full:.1f} h per seed", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
