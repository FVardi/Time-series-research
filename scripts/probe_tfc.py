"""Diagnostic: frozen-encoder probes on our pretrained TF-C encoders (D45).

Question: did pretraining on FD-A learn features that separate the FD-B classes,
independently of the short fine-tuning? The pretrained encoders of a finished
scripts/tfc_ours.py run are frozen; the harness probes (D26) are trained on the
FD-B training windows and scored once on the FD-B test windows.

Usage:
    uv run python scripts/probe_tfc.py configs/probe_tfc_fd.yaml

Writes into runs/<date_time>_<name>/: results.csv (one row per encoder, seed,
label fraction and probe) and summary.csv (mean and standard deviation over seeds).
Representations are not stored.
"""

import os
import sys

import pandas as pd
import torch

from harness import evaluate, runs, transfer
from harness.data import canonical
from harness.methods import tfc


def check_settings(ev):
    """Stop if a setting is still an open placeholder (null in the YAML)."""
    for k, v in ev.items():
        if v is None:
            raise ValueError(f"evaluation.{k} is not set (placeholder); see D45")


def main(config_path):
    cfg = runs.load_config(config_path)
    ev = cfg["evaluation"]
    check_settings(ev)
    source = cfg["tfc_run"]  # folder of the finished tfc_ours run whose encoders are probed
    tcfg = runs.load_config(os.path.join(source, "config.yaml"))  # data and model settings of that run
    folder = runs.new_run_dir(config_path, cfg)

    # FD-B training and test windows, exactly as in the TF-C run.
    meta = canonical.load_metadata(tcfg["data_root"], tcfg["dataset"])
    X_tr, y_tr, _ = transfer.load_set(tcfg, meta, tcfg["target_run"], "train")
    X_te, y_te, _ = transfer.load_set(tcfg, meta, tcfg["target_run"], "test")
    s = tcfg["tfc"]
    length, n_channels = X_tr.shape[1], X_tr.shape[2]

    rows = []
    for seed in tcfg["seeds"]:
        encoders = {}
        # The pretrained encoder of this seed, frozen.
        model = tfc.TFC(length, n_channels, s["encoder"])
        model.load_state_dict(torch.load(os.path.join(source, f"seed_{seed}", "pretrained.pt"),
                                         map_location="cpu", weights_only=True))
        encoders["pretrained"] = model
        if ev["random_init_reference"]:
            # Same architecture, not trained: initialised with the same seed (D45 reference).
            runs.set_seed(seed)
            encoders["random_init"] = tfc.TFC(length, n_channels, s["encoder"])

        for name, model in encoders.items():
            model = model.to(cfg["device"])
            Z_tr = tfc.encode(model, X_tr, s["eval_chunk"], cfg["device"])  # [z_T; z_F] per window
            Z_te = tfc.encode(model, X_te, s["eval_chunk"], cfg["device"])
            for fraction in ev["label_fractions"]:
                idx = evaluate.label_subset(y_tr, fraction, seed)  # D29
                for probe in ev["probes"]:
                    clf = evaluate.PROBES[probe](Z_tr[idx], y_tr[idx])
                    rows.append({"encoder": name, "seed": seed, "label_fraction": fraction,
                                 "n_labels": len(idx), "probe": probe, **evaluate.scores(clf, Z_te, y_te)})
                    print(rows[-1], flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(folder, "results.csv"), index=False)
    summary = results.groupby(["encoder", "probe", "label_fraction"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(folder, "summary.csv"))
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1])
