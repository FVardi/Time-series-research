"""Timing test only (not a run, nothing saved in runs/): TS-JEPA on Paderborn combination 0,
a few epochs in fp32 and in mixed precision (bf16 autocast), to estimate the full run time.

Data and inputs as tfc_paderborn.yaml (D220, D222, D225); TS-JEPA settings as tsjepa_cwru.yaml,
except the number of epochs. The harness code is not changed: mixed precision is switched on by
wrapping the unchanged training function in torch.autocast.

Run from the repo root:
    uv run python "Claude outputs/time_tsjepa_amp.py"
"""
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, "scripts")  # to import windowed_inputs from scripts/pretrain.py
import pretrain  # noqa: E402
from harness import protocols, runs  # noqa: E402
from harness.data import canonical  # noqa: E402
from harness.methods import tsjepa  # noqa: E402

EPOCHS = 4  # per precision; the first epoch is treated as warm-up and not counted

cfg = runs.load_config("configs/tfc_paderborn.yaml")  # data part only
cfg["split"] = protocols.PROTOCOLS["paderborn_lessmeier"](0)  # combination 0
s = runs.load_config("configs/tsjepa_cwru.yaml")["tsjepa"]  # method settings
s["epochs"] = EPOCHS
s["print_every"] = 1

meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
with tempfile.TemporaryDirectory() as tmp:  # windowed_inputs writes standardisation.json here
    X_pre, *_ = pretrain.windowed_inputs(cfg, meta, tmp)
batches = -(-len(X_pre) // s["batch_size"])
print(f"{len(X_pre)} pretraining windows, {batches} batches per epoch")


def time_epochs(mixed):
    """Seconds per epoch (epochs 2.. only), timed with one callback after each epoch."""
    stamps = []

    def on_epoch(done, encoder):  # called before epoch 1 and after every epoch; changes nothing
        torch.cuda.synchronize()
        stamps.append(time.perf_counter())

    runs.set_seed(0)
    if mixed:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, log = tsjepa.pretrain(X_pre, s, "cuda", on_epoch=on_epoch)
    else:
        _, log = tsjepa.pretrain(X_pre, s, "cuda", on_epoch=on_epoch)
    per_epoch = (stamps[-1] - stamps[1]) / (len(stamps) - 2)
    return per_epoch, log


results = {}
for mixed in (False, True):
    name = "bf16 mixed" if mixed else "fp32"
    print(f"--- {name}")
    per_epoch, log = time_epochs(mixed)
    results[name] = per_epoch
    print(f"{name}: {per_epoch:.2f} s per epoch, losses {[round(l, 4) for l in log]}")

print("\nEstimated pretraining time, Paderborn, all 10 combinations:")
for name, t in results.items():
    for epochs in (5000, 2000):
        print(f"  {name:10s} {epochs:5d} epochs: {t * epochs / 3600:6.1f} h per combination, "
              f"{10 * t * epochs / 3600:6.1f} h for 10")
