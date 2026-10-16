"""Choose TS-JEPA's learning rate on a validation split of the training series (D67).

Usage:
    uv run python scripts/search_tsjepa_lr.py configs/tsjepa_forda_lr_search.yaml

Only the official training split is loaded; the test split is never read. The training
series are split 80/20 (stratified by class). For each learning rate: pretrain on the 80 %
without labels (all other settings from base_config), encode both parts, fit the
logistic-regression probe on the 80 % with all labels, score it on the 20 %.

Writes into runs/<date_time>_<name>/:
    base_config.yaml        copy of the TS-JEPA config used
    split.csv               which training capture is in 'fit' (80 %) or 'val' (20 %)
    loss_lr_<value>.json    pretraining loss per epoch, per learning rate
    results.csv             one row per learning rate (written after each, so partial results survive)
and prints the chosen learning rate: the highest validation accuracy (if tied, the first in the list).
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from harness import evaluate, runs
from harness.data import canonical
from harness.methods import tsjepa


def main(config_path):
    cfg = runs.load_config(config_path)
    base = runs.load_config(cfg["base_config"])
    if cfg["selection"] != "logreg_accuracy":
        raise ValueError(f"unknown selection {cfg['selection']!r}")
    s = base["tsjepa"]
    s["optimizer"] = dict(s["optimizer"], lr=0.0)  # replaced per learning rate below; checked for other placeholders
    tsjepa.check_settings(s)
    folder = runs.new_run_dir(config_path, cfg)
    shutil.copy(cfg["base_config"], os.path.join(folder, "base_config.yaml"))

    # Training series only, split 80/20 (stratified).
    meta = canonical.load_metadata(base["data_root"], base["dataset"])
    caps = meta["captures"][meta["captures"].official_split == "train"]
    X = canonical.load_captures(base["data_root"], base["dataset"], base["run_id"], caps.capture_id, base["channels"])
    y = caps.label.to_numpy()
    fit, val = train_test_split(np.arange(len(y)), test_size=cfg["validation_fraction"], stratify=y,
                                random_state=cfg["split_seed"])
    fit, val = np.sort(fit), np.sort(val)
    pd.DataFrame({"capture_id": caps.capture_id.to_numpy(),
                  "part": np.where(np.isin(np.arange(len(y)), val), "val", "fit")}).to_csv(
        os.path.join(folder, "split.csv"), index=False)
    print(f"{len(fit)} series for pretraining and the probe, {len(val)} for validation", flush=True)

    rows = []
    for lr in cfg["learning_rates"]:
        print(f"learning rate {lr:g}: pretraining", flush=True)
        s["optimizer"]["lr"] = lr
        runs.set_seed(cfg["seed"])
        model, loss_log = tsjepa.pretrain(X[fit], s, base["device"])
        runs.save_json(os.path.join(folder, f"loss_lr_{lr:g}.json"), loss_log)
        Z_fit, Z_val = tsjepa.encode(model, X[fit]), tsjepa.encode(model, X[val])
        clf = evaluate.logreg(Z_fit, y[fit])  # D26, D29
        rows.append({"lr": lr, **evaluate.scores(clf, Z_val, y[val]), "final_loss": loss_log[-1]})
        pd.DataFrame(rows).to_csv(os.path.join(folder, "results.csv"), index=False)
        print(f"learning rate {lr:g}: validation accuracy {rows[-1]['accuracy']:.4f}", flush=True)

    results = pd.DataFrame(rows)
    best = results.iloc[int(np.argmax(results.accuracy.to_numpy()))]  # first in the list if tied
    print(results.to_string(index=False))
    print(f"Chosen learning rate (D67): {best.lr:g} (validation accuracy {best.accuracy:.4f})")


if __name__ == "__main__":
    main(sys.argv[1])
