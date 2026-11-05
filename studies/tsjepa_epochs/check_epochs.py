"""TS-JEPA epoch diagnostic (D200): does TS-JEPA need its 5,000 epochs (D66)?

Usage:
    uv run python studies/tsjepa_epochs/check_epochs.py studies/tsjepa_epochs/configs/epochs_cwru.yaml

Trains TS-JEPA exactly as the base config's run (same data, split, standardisation, settings and
seed; scripts/pretrain.py's windowed_inputs and harness/methods/tsjepa.py, unchanged) and, before
the first epoch and every 'every' epochs, measures the encoder:
  rankme       - RankMe of the representations of all training windows (as D152)
  val_accuracy - the standard logreg probe (harness/evaluate.py), fitted with all labels on 80 % of
                 the training windows, scored on the other 20 % (validation set: stratified by class,
                 one fixed draw). The test windows are never used.
The measurements put the random number generators back as they found them, so training is the same
as without them (tests/test_tsjepa.py). Run it with the GPU to itself, or the times are not comparable.
Writes runs/<date_time>_<name>/epochs.csv (one row per measurement, written as the run goes),
loss.json, timing.json, standardisation.json, hardware.json.
"""

import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from harness import diagnostics, evaluate, runs
from harness.data import canonical
from harness.methods import tsjepa
from scripts.pretrain import clock, hardware, windowed_inputs


def rng_state():
    """The states of every random number generator training uses (Python, NumPy, PyTorch, CUDA)."""
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(state):
    py, nps, tor, cuda = state
    random.setstate(py)
    np.random.set_state(nps)
    torch.set_rng_state(tor)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def measurer(cfg, X_train, y_train, fit, val, epochs, rows, folder):
    """The on_epoch function for tsjepa.pretrain: measures every cfg['every'] epochs (and before training)."""
    def measure(done, encoder):
        if done % cfg["every"] and done != epochs:
            return
        start, state = time.perf_counter(), rng_state()
        Z = tsjepa.encode(encoder, X_train)  # evaluation mode, no gradients; training resumes in train mode
        rank = diagnostics.rankme(torch.from_numpy(Z.astype(np.float64)), cfg["rankme_epsilon"])
        probe = evaluate.logreg(Z[fit], y_train[fit])
        accuracy = float((probe.predict(Z[val]) == y_train[val]).mean())
        set_rng_state(state)
        rows.append({"epoch": done, "rankme": rank, "val_accuracy": accuracy, "measure_s": time.perf_counter() - start})
        pd.DataFrame(rows).to_csv(os.path.join(folder, "epochs.csv"), index=False)  # kept up to date while running
        print(f"  epoch {done}: RankMe {rank:.1f}, validation accuracy {accuracy:.3f}", flush=True)
    return measure


def main(config_path):
    cfg = runs.load_config(config_path)
    base = runs.load_config(cfg["base_config"])
    folder = runs.new_run_dir(config_path, cfg)
    runs.save_json(os.path.join(folder, "hardware.json"), hardware(base["device"]))
    meta = canonical.load_metadata(base["data_root"], base["dataset"])
    X_pre, X_train, labels_train, _, _, _ = windowed_inputs(base, meta, folder)  # test windows not used
    y_train = np.searchsorted(np.unique(labels_train), labels_train)
    fit, val = train_test_split(np.arange(len(X_train)), test_size=cfg["validation_fraction"], stratify=y_train,
                                random_state=cfg["validation_random_state"])
    print(f"validation: {len(val)} of the {len(X_train)} training windows; probe fitted on {len(fit)}", flush=True)

    rows = []
    s = base["tsjepa"]
    runs.set_seed(cfg["seed"])  # as scripts/pretrain.py, right before pretraining
    t0 = clock(base["device"])
    _, loss = tsjepa.pretrain(X_pre, s, base["device"],
                              on_epoch=measurer(cfg, X_train, y_train, fit, val, s["epochs"], rows, folder))
    total = clock(base["device"]) - t0
    measuring = sum(r["measure_s"] for r in rows)
    runs.save_json(os.path.join(folder, "timing.json"), {"pretrain_s": total - measuring, "measure_s": measuring})
    runs.save_json(os.path.join(folder, "loss.json"), [float(x) for x in loss])
    print(f"Done: {folder}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    main(sys.argv[1])
