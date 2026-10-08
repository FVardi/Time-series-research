"""TF-C, our implementation (D41), on a transfer setting such as FD-A -> FD-B.

Usage:
    uv run python scripts/tfc_ours.py configs/tfc_fd.yaml

Per seed: pretrain on the source training set, fine-tune on the target training set,
keep the epoch with the best target-validation macro-F1, score the target test set
once. Writes into runs/<date_time>_<name>/:
    windows.csv                   which windows were used (D33)
    seed_<k>/pretrain_loss.csv    pretraining loss per epoch
    seed_<k>/epochs.csv           fine-tuning loss and validation scores per epoch
    seed_<k>/pretrained.pt        pretrained weights (for probes later)
    results.csv                   test scores per seed
    summary.csv                   mean and standard deviation over seeds
and prints the check against the published accuracy (D38, criterion as D26).
"""

import os
import sys

import pandas as pd

from harness import runs, transfer
from harness.methods import tfc


def main(config_path):
    cfg = runs.load_config(config_path)
    tfc.check_settings(cfg["tfc"])  # stop before any work if a placeholder is still open
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
        runs.set_seed(seed)
        print(f"seed {seed}: pretraining on {len(data['source_train'][0])} windows", flush=True)
        model, losses = tfc.pretrain(data["source_train"][0], cfg["tfc"], cfg["device"])
        losses.to_csv(os.path.join(out, "pretrain_loss.csv"), index=False)
        tfc.save(model, os.path.join(out, "pretrained.pt"))

        print(f"seed {seed}: fine-tuning on {len(data['target_train'][0])} windows", flush=True)
        epochs, test, best_epoch = tfc.finetune(model, data, cfg["tfc"], cfg["device"])
        epochs.to_csv(os.path.join(out, "epochs.csv"), index=False)
        results.append({"seed": seed, "epoch": best_epoch, **test})
        print(f"seed {seed}: epoch {best_epoch}, test accuracy {test['accuracy']:.4f}, macro-F1 {test['f1']:.4f}")

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(folder, "results.csv"), index=False)
    metrics = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    summary = results[metrics].agg(["mean", "std"])
    summary.to_csv(os.path.join(folder, "summary.csv"))
    print(summary)

    acc = results.accuracy
    mean, sd, pub = acc.mean(), acc.std(), cfg["evaluation"]["published_accuracy"]
    ok = abs(pub - mean) <= max(2 * sd, 0.01)
    print(f"Published TF-C: ours {mean:.4f} ± {sd:.4f} (n={len(acc)}), published {pub}: {'WITHIN' if ok else 'OUTSIDE'}")


if __name__ == "__main__":
    main(sys.argv[1])
