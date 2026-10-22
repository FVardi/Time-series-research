"""TF-C study, part 2: ablation by building blocks (D95, D96, D97).

Usage (from the repo root), one stage per config:
    uv run python studies/tfc_ablation/ablation.py studies/tfc_ablation/configs/ablation_forda_A.yaml

Stages, all built from our TF-C (harness/methods/tfc.py, D41, D42):
    A  time branch only            L_T
    B  frequency branch only       L_F
    C  both branches, uncoupled    L_T + L_F
    D  both, plain alignment       λ(L_T + L_F) + (1 − λ) S_TF        (λ as in E, D97)
    E  full TF-C                   λ(L_T + L_F) + (1 − λ) L_C          (exactly tfc.pretrain)
Variants of D and E (D101), to find what in L_C causes the collapse of z:
    F  E without the pair terms     λ(L_T + L_F) + 3(1 − λ) S_TF      (S_TF with the weight it has in E)
    G  hinged L_C                   λ(L_T + L_F) + (1 − λ) Σ max(0, S_TF − S_pair + δ)
    H  augmented pairs as positives λ(L_T + L_F) + (1 − λ) (S_TF + S_pair,1 + S_pair,2 + S_pair,3) / 4
S_pair,1..3 are the NT-Xent terms of L_C: (z_T, z̃_F), (z̃_T, z_F), (z̃_T, z̃_F). F, G and H are our
constructions (not in the paper or the official code). With both branches, all of S_TF, the three
S_pair and L_C are logged in every stage (C-H), whether or not they are trained.
In A and B only the one branch is built and trained (D97). In C the projectors exist but
receive no gradient, so they keep their initial weights.

Per seed: pretraining with the collapse diagnostics after every epoch (D93, D94; as in
collapse.py), then frozen probes on the representations listed in the config, trained
on label subsets of the training split and scored once on the test split (D26, D29, D40).

Writes into runs/<date_time>_<name>/:
    windows.csv                 pretraining, diagnostic and test windows (D33)
    seed_<k>/pretrain_loss.csv  per epoch: total loss and its parts
    seed_<k>/collapse.csv       per epoch and representation: the measures of D93
    seed_<k>/spectra.csv        per epoch and representation: covariance eigenvalues
    seed_<k>/pretrained.pt      weights after the last epoch
    probes.csv                  per seed, representation, label fraction and probe: accuracy, macro-F1
    summary.csv                 mean and standard deviation over seeds
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from harness import evaluate, runs, transfer
from harness.data import canonical
from harness.methods import tfc
from studies.tfc_ablation.collapse import Monitor, check_diagnostics, diagnostic_rows, represent

# Representations each stage produces (D96: A-C are evaluated on h, D-E on h and z; D101: F-H as D-E).
BOTH = {"h_t", "h_f", "h", "z_t", "z_f", "z"}
AVAILABLE = {"A": {"h_t", "z_t"}, "B": {"h_f", "z_f"},
             "C": BOTH, "D": BOTH, "E": BOTH, "F": BOTH, "G": BOTH, "H": BOTH}


def build_model(stage, length, channels, e):
    """A: time branch only; B: frequency branch only; C-H: the full TF-C model (both branches)."""
    if stage == "A":
        return nn.ModuleDict({"time": tfc.Branch(length, channels, e)})
    if stage == "B":
        return nn.ModuleDict({"freq": tfc.Branch(length, channels, e)})
    return tfc.TFC(length, channels, e)


def stage_loss(stage, model, x, s):
    """Loss of one batch for the given stage, and its parts (floats, for the log)."""
    l = s["loss"]
    tau, lam = l["temperature"], l["lambda"]
    if stage == "A":  # time view vs its augmented version
        h, _ = model.time(x)
        h_aug, _ = model.time(tfc.time_augment(x, s["time_aug"]))
        loss = tfc.nt_xent(h, h_aug, tau)
        return loss, {"loss_t": loss.item()}
    if stage == "B":  # frequency view vs its augmented version
        xf = tfc.spectrum(x)
        h, _ = model.freq(xf)
        h_aug, _ = model.freq(tfc.freq_augment(xf, s["freq_aug"]["n_components"], s["freq_aug"]["alpha"]))
        loss = tfc.nt_xent(h, h_aug, tau)
        return loss, {"loss_f": loss.item()}

    out, out_aug = tfc.augmented_step(model, x, s)  # same random draws as tfc.pretrain
    (h_t, z_t, h_f, z_f), (h_ta, z_ta, h_fa, z_fa) = out, out_aug
    loss_t, loss_f = tfc.nt_xent(h_t, h_ta, tau), tfc.nt_xent(h_f, h_fa, tau)
    s_tf = tfc.nt_xent(z_t, z_f, tau)
    # The three cross-domain pairs with augmented views, as in L_C: S_pair,1..3
    s_pair = [tfc.nt_xent(a, b, tau) for a, b in ((z_t, z_fa), (z_ta, z_f), (z_ta, z_fa))]
    delta = l["delta"]
    if stage == "C":
        loss = loss_t + loss_f
    elif stage == "D":
        loss = lam * (loss_t + loss_f) + (1 - lam) * s_tf
    elif stage == "E":  # the full TF-C loss, computed by the harness function itself
        loss = tfc.tfc_loss(out, out_aug, l)
    elif stage == "F":  # E without the S_pair terms: L_C = 3 S_TF - sum(S_pair) + 3 delta, so S_TF has weight 3(1 - lam) in E
        loss = lam * (loss_t + loss_f) + 3 * (1 - lam) * s_tf
    elif stage == "G":  # L_C with a hinge on each of its three terms
        loss = lam * (loss_t + loss_f) + (1 - lam) * sum(torch.clamp(s_tf - p + delta, min=0) for p in s_pair)
    elif stage == "H":  # the augmented cross-domain pairs as further positives, mean of the four alignment terms
        loss = lam * (loss_t + loss_f) + (1 - lam) * (s_tf + sum(s_pair)) / 4
    else:
        raise ValueError(f"unknown stage {stage}")
    loss_c = sum(s_tf - p + delta for p in s_pair)  # L_C for the log, in every stage with both branches
    return loss, {"loss_t": loss_t.item(), "loss_f": loss_f.item(), "s_tf": s_tf.item(), "loss_c": loss_c.item(),
                  **{f"s_pair_{i}": p.item() for i, p in enumerate(s_pair, 1)}}


def pretrain(stage, X, X_diag, s, d, device):
    """Pretraining of one stage with the Monitor (epoch 0 = untrained model). Same loop as tfc.pretrain."""
    tfc.check_settings(s)
    check_diagnostics(d)
    x_all = tfc.to_device(X, device)
    model = build_model(stage, x_all.shape[2], x_all.shape[1], s["encoder"]).to(device)
    monitor = Monitor(tfc.to_device(X_diag, device), s, d, device)
    monitor.measure(model, 0)
    p = s["pretrain"]
    opt = tfc.adam(model.parameters(), p)
    log = []
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        parts = []
        for idx in tfc.batches(len(x_all), p["batch_size"], p["drop_last"]):
            loss, part = stage_loss(stage, model, x_all[idx.to(device)], s)
            opt.zero_grad()  # every step
            loss.backward()
            opt.step()
            parts.append({"loss": loss.item(), **part})
        log.append({"epoch": epoch, **pd.DataFrame(parts).mean().to_dict()})
        print(f"  pretrain epoch {epoch}: " + ", ".join(f"{k} {v:.4f}" for k, v in log[-1].items() if k != "epoch"), flush=True)
        monitor.measure(model, epoch)
    return model, pd.DataFrame(log), monitor


def probe(model, X_train, y_train, X_test, y_test, names, ev, seed, chunk, device):
    """Frozen probes on each representation in names (D26 probes, D40 label budgets; test set scored once)."""
    x_tr, x_te = tfc.to_device(X_train, device), tfc.to_device(X_test, device)
    R_tr = represent(model, x_tr, tfc.spectrum(x_tr), chunk)
    R_te = represent(model, x_te, tfc.spectrum(x_te), chunk)
    rows = []
    for name in names:
        Z_tr, Z_te = R_tr[name].double().cpu().numpy(), R_te[name].double().cpu().numpy()
        for fraction in ev["label_fractions"]:
            idx = evaluate.label_subset(y_train, fraction, seed)
            for p in ev["probes"]:
                clf = evaluate.PROBES[p](Z_tr[idx], y_train[idx])
                rows.append({"seed": seed, "representation": name, "label_fraction": fraction, "n_labels": len(idx),
                             "probe": p, **evaluate.scores(clf, Z_te, y_test)})
                print(f"  probe {name} {p} {fraction}: accuracy {rows[-1]['accuracy']:.4f}", flush=True)
    return rows


def main(config_path):
    cfg = runs.load_config(config_path)
    stage, ev = cfg["stage"], cfg["evaluation"]
    tfc.check_settings(cfg["tfc"])  # stop before any work if a placeholder is still open
    check_diagnostics(cfg["diagnostics"])
    for key, names in (("diagnostics.representations", cfg["diagnostics"]["representations"]),
                       ("evaluation.representations", ev["representations"])):
        missing = set(names) - AVAILABLE[stage]
        if missing:
            raise ValueError(f"{key}: stage {stage} has no {sorted(missing)}")
    if cfg["seeds"] is None:
        raise ValueError("seeds is not set (placeholder)")
    folder = runs.new_run_dir(config_path, cfg)

    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    X, labels, ids = transfer.load_set(cfg, meta, cfg["run_id"], cfg["train_split"])
    X_test, labels_test, ids_test = transfer.load_set(cfg, meta, cfg["run_id"], cfg["test_split"])
    classes = np.unique(labels)  # class labels -> 0..K-1, ordered as in the training set (as scripts/pretrain.py)
    y, y_test = np.searchsorted(classes, labels), np.searchsorted(classes, labels_test)
    rows = diagnostic_rows(len(X), cfg["diagnostics"]["windows"], cfg["diagnostics"]["seed"])

    def table(split, name, r, cap_ids, n):
        return pd.DataFrame({"split": split, "set": name, "row": np.arange(len(r)), "run_id": cfg["run_id"],
                             "capture_id": cap_ids[r], "channels": ";".join(cfg["channels"]), "start": 0, "stop": n})
    windows = pd.concat([table("train", "pretrain", np.arange(len(X)), ids, X.shape[1]),
                         table("train", "diagnostics", rows, ids, X.shape[1]),
                         table("test", "probe_test", np.arange(len(X_test)), ids_test, X_test.shape[1])], ignore_index=True)
    runs.check_no_overlap(windows)  # D33
    windows.to_csv(os.path.join(folder, "windows.csv"), index=False)

    results = []
    for seed in cfg["seeds"]:
        out = os.path.join(folder, f"seed_{seed}")
        os.makedirs(out)
        runs.set_seed(seed)
        print(f"seed {seed}: stage {stage}, pretraining on {len(X)} windows, measuring {len(rows)}", flush=True)
        model, losses, monitor = pretrain(stage, X, X[rows], cfg["tfc"], cfg["diagnostics"], cfg["device"])
        losses.to_csv(os.path.join(out, "pretrain_loss.csv"), index=False)
        pd.DataFrame(monitor.rows).to_csv(os.path.join(out, "collapse.csv"), index=False)
        if monitor.spectra:
            pd.concat(monitor.spectra).to_csv(os.path.join(out, "spectra.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(out, "pretrained.pt"))
        results += probe(model, X, y, X_test, y_test, ev["representations"], ev, seed, cfg["tfc"]["eval_chunk"], cfg["device"])
        pd.DataFrame(results).to_csv(os.path.join(folder, "probes.csv"), index=False)  # written after every seed

    summary = (pd.DataFrame(results).groupby(["representation", "probe", "label_fraction"])[["accuracy", "macro_f1"]]
               .agg(["mean", "std"]))
    summary.to_csv(os.path.join(folder, "summary.csv"))
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1])
