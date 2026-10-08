"""Reproduce TF-C on FD-A -> FD-B with the official code (D35 step 1, D36-D39).

Usage:
    uv run python scripts/tfc_official.py configs/tfc_official_fd.yaml

Per seed: pretrain on FD-A (train), fine-tune on FD-B (train), score FD-B test and
validation after every fine-tuning epoch. Writes into runs/<date_time>_<name>/:
    windows.csv                   which windows were used (D33)
    seed_<k>/pretrain_loss.csv    pretraining loss per epoch
    seed_<k>/epochs.csv           fine-tuning loss and test/validation scores per epoch
    results.csv                   per seed, the two protocols of D37 (D43); diverged seeds marked (D44)
    summary.csv                   mean and standard deviation over the seeds that did not diverge
and prints the reproduction check (D38).
"""

import os
import sys

import numpy as np
import pandas as pd

from harness import runs, transfer
from harness.methods import tfc_official


def pick(epochs, by):
    """Row of the epoch with the highest value in column `by` (first one if tied, as np.argmax)."""
    return epochs.iloc[int(np.argmax(epochs[by].to_numpy()))]


def main(config_path):
    cfg = runs.load_config(config_path)
    folder = runs.new_run_dir(config_path, cfg)

    sets = transfer.load_sets(cfg)
    windows = transfer.window_table(cfg, sets)
    runs.check_no_overlap(windows)  # D33
    windows.to_csv(os.path.join(folder, "windows.csv"), index=False)
    data = {name: (X, y) for name, (X, y, _) in sets.items()}

    results = []
    for seed in cfg["seeds"]:
        out = os.path.join(folder, f"seed_{seed}")
        os.makedirs(out)
        print(f"seed {seed}: pretraining on {len(data['source_train'][0])} windows", flush=True)
        weights, losses, diverged = tfc_official.pretrain(data, cfg["tfc_official"], cfg["device"], seed)
        losses.to_csv(os.path.join(out, "pretrain_loss.csv"), index=False)
        # The pretrained weights (about 3.4 GB per seed) are not saved.
        if diverged:  # D44: recorded as a result, no fine-tuning; the other seeds continue
            for protocol in ("official", "clean"):
                results.append({"seed": seed, "protocol": protocol, "diverged": True, "epoch": None,
                                "accuracy": np.nan, "macro_f1": np.nan, "auroc": np.nan, "auprc": np.nan})
            continue

        print(f"seed {seed}: fine-tuning on {len(data['target_train'][0])} windows", flush=True)
        epochs = tfc_official.finetune_and_test(weights, data, cfg["tfc_official"], cfg["device"], seed)
        epochs.to_csv(os.path.join(out, "epochs.csv"), index=False)

        # D37: (1) official protocol = epoch with the best test accuracy;
        #      (2) clean protocol = epoch with the best validation macro-F1 (D43, as our TF-C, D41).
        for protocol, by in (("official", "test_accuracy"), ("clean", "val_f1")):
            row = pick(epochs, by)
            results.append({"seed": seed, "protocol": protocol, "diverged": False, "epoch": int(row.epoch),
                            "accuracy": row.test_accuracy, "macro_f1": row.test_f1,
                            "auroc": row.test_auroc, "auprc": row.test_auprc})

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(folder, "results.csv"), index=False)
    # Mean and SD over the seeds that did not diverge (D44); the count is reported alongside.
    ok_runs = results[~results.diverged]
    summary = ok_runs.groupby("protocol")[["accuracy", "macro_f1", "auroc", "auprc"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(folder, "summary.csv"))
    n_div = int(results[results.protocol == "official"].diverged.sum())
    print(f"Diverged in pretraining: {n_div} of {len(cfg['seeds'])} seeds")
    print(summary)
    if n_div == len(cfg["seeds"]):
        print("Reproduction: no seed finished pretraining, nothing to compare.")
        return

    # Reproduction check (D38, criterion as D26): official protocol vs the published accuracy.
    acc = ok_runs[ok_runs.protocol == "official"].accuracy
    mean, sd, pub = acc.mean(), acc.std(), cfg["evaluation"]["published_accuracy"]
    ok = abs(pub - mean) <= max(2 * sd, 0.01)
    print(f"Reproduction: ours {mean:.4f} ± {sd:.4f} (n={len(acc)}), published {pub}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
