"""TF-C study, part 1: examination of collapse during pretraining (D72, D73).

Usage (from the repo root):
    uv run python studies/tfc_ablation/collapse.py studies/tfc_ablation/configs/collapse_forda.yaml

Pretrains our TF-C (harness/methods/tfc.py, D41, D42) exactly as tfc.pretrain does, and
before training (epoch 0) and after every epoch measures the representations of a fixed
set of windows (D73, harness/diagnostics.py). The training is the same as tfc.pretrain
with the same seed: the measurements run in evaluation mode and use their own random
numbers (tested in tests/test_collapse_study.py).

Writes into runs/<date_time>_<name>/:
    windows.csv                 pretraining and diagnostic windows (D33)
    seed_<k>/pretrain_loss.csv  per epoch: total loss and its parts L_T, L_F, L_C, S_TF (means over batches)
    seed_<k>/collapse.csv       per epoch and representation: the measures of D73
    seed_<k>/spectra.csv        per epoch and representation: all covariance eigenvalues (k = 1 is the largest)
    seed_<k>/pretrained.pt      weights after the last epoch
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

from harness import diagnostics, runs, transfer
from harness.data import canonical
from harness.methods import tfc

REPRESENTATIONS = ("h_t", "h_f", "z_t", "z_f", "z")  # z = [z_T; z_F], the representation for probes (tfc.encode)


def check_diagnostics(d):
    """Stop if a diagnostics setting is still an open placeholder (null), or a representation name is unknown."""
    def walk(x, path):
        for k, v in x.items():
            if isinstance(v, dict):
                walk(v, f"{path}{k}.")
            elif v is None:
                raise ValueError(f"diagnostics.{path}{k} is not set (placeholder); see D73")
    walk(d, "")
    unknown = set(d["representations"]) - set(REPRESENTATIONS)
    if unknown:
        raise ValueError(f"unknown representations {sorted(unknown)}; choose from {REPRESENTATIONS}")


def diagnostic_rows(n, windows, seed):
    """Which pretraining windows are measured: all of them, or a fixed random subset (same for every seed and epoch)."""
    if windows == "all":
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=windows, replace=False))  # own generator: global RNG untouched


def represent(model, x_t, x_f, chunk):
    """h_T, h_F, z_T, z_F and [z_T; z_F] for time inputs x_t and frequency inputs x_f, in evaluation mode."""
    model.eval()
    parts = {name: [] for name in REPRESENTATIONS}
    with torch.no_grad():
        for i in range(0, len(x_t), chunk):
            h_t, z_t, h_f, z_f = model(x_t[i:i + chunk], x_f[i:i + chunk])
            for name, value in zip(("h_t", "z_t", "h_f", "z_f", "z"), (h_t, z_t, h_f, z_f, torch.cat([z_t, z_f], 1))):
                parts[name].append(value)
    return {name: torch.cat(v) for name, v in parts.items()}


def fixed_augmented_views(x, s, seed, device):
    """One augmented time view and one augmented frequency view of the diagnostic windows, drawn once.

    Uses the same augmentations as pretraining (tfc.time_augment, tfc.freq_augment), seeded with
    the diagnostics seed. The global random states (PyTorch CPU/GPU, NumPy) are saved and restored,
    so the pretraining is not affected.
    """
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        np.random.seed(seed)  # tfc.permutation draws with NumPy
        x_t = tfc.time_augment(x, s["time_aug"])
        x_f = tfc.freq_augment(tfc.spectrum(x), s["freq_aug"]["n_components"], s["freq_aug"]["alpha"])
    np.random.set_state(np_state)
    return x_t, x_f


class Monitor:
    """Measures the diagnostic windows after every epoch and collects the rows for collapse.csv and spectra.csv."""

    def __init__(self, x_diag, s, d, device):
        self.x, self.xf = x_diag, tfc.spectrum(x_diag)
        self.s, self.d = s, d
        self.aug = fixed_augmented_views(x_diag, s, d["seed"], device) if d["alignment"] else None
        self.generator = torch.Generator(device=device)  # SIGReg directions: own random numbers
        self.generator.manual_seed(d["seed"])
        self.directions = {}  # per representation, if the same directions are used every epoch
        self.rows, self.spectra = [], []

    def directions_for(self, name, dim):
        if not self.d["sigreg"]["same_directions_every_epoch"] or name not in self.directions:
            self.directions[name] = diagnostics.sigreg_directions(dim, self.d["sigreg"]["n_slices"], self.generator)
        return self.directions[name]

    def measure(self, model, epoch):
        reps = represent(model, self.x, self.xf, self.s["eval_chunk"])
        reps_aug = represent(model, *self.aug, self.s["eval_chunk"]) if self.aug else {}
        for name in self.d["representations"]:
            Z = reps[name]
            row, eig = diagnostics.measures(Z, self.d, self.directions_for(name, Z.shape[1]), reps_aug.get(name))
            self.rows.append({"epoch": epoch, "representation": name, **row})
            if eig is not None:
                self.spectra.append(pd.DataFrame({"epoch": epoch, "representation": name,
                                                  "k": np.arange(1, len(eig) + 1), "eigenvalue": eig}))
        h = {r["representation"]: r for r in self.rows if r["epoch"] == epoch}
        print(f"  epoch {epoch}: effective rank " + ", ".join(f"{k} {v['rankme']:.1f}" for k, v in h.items()), flush=True)


def pretrain(X, X_diag, s, d, device):
    """tfc.pretrain with the Monitor added (epoch 0 = untrained model). Returns model, loss table, monitor."""
    tfc.check_settings(s)
    check_diagnostics(d)
    x_all = tfc.to_device(X, device)
    model = tfc.TFC(x_all.shape[2], x_all.shape[1], s["encoder"]).to(device)  # same order of random draws as tfc.pretrain
    monitor = Monitor(tfc.to_device(X_diag, device), s, d, device)
    monitor.measure(model, 0)
    p, l = s["pretrain"], s["loss"]
    opt = tfc.adam(model.parameters(), p)
    log = []
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        parts = []
        for idx in tfc.batches(len(x_all), p["batch_size"], p["drop_last"]):
            out, out_aug = tfc.augmented_step(model, x_all[idx.to(device)], s)
            loss = tfc.tfc_loss(out, out_aug, l)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():  # the parts of the loss, for the log only (no random numbers used)
                (h_t, z_t, h_f, z_f), (h_ta, z_ta, h_fa, z_fa) = out, out_aug
                tau = l["temperature"]
                s_tf = tfc.nt_xent(z_t, z_f, tau)
                loss_c = sum(s_tf - tfc.nt_xent(a, b, tau) + l["delta"] for a, b in ((z_t, z_fa), (z_ta, z_f), (z_ta, z_fa)))
                parts.append([loss.item(), tfc.nt_xent(h_t, h_ta, tau).item(), tfc.nt_xent(h_f, h_fa, tau).item(),
                              float(loss_c), s_tf.item()])
        m = np.mean(parts, axis=0)
        log.append({"epoch": epoch, "loss": m[0], "loss_t": m[1], "loss_f": m[2], "loss_c": m[3], "s_tf": m[4]})
        print(f"  pretrain epoch {epoch}: loss {m[0]:.4f} (L_T {m[1]:.4f}, L_F {m[2]:.4f}, L_C {m[3]:.4f})", flush=True)
        monitor.measure(model, epoch)
    return model, pd.DataFrame(log), monitor


def main(config_path):
    cfg = runs.load_config(config_path)
    tfc.check_settings(cfg["tfc"])  # stop before any work if a placeholder is still open
    check_diagnostics(cfg["diagnostics"])
    if cfg["seeds"] is None:
        raise ValueError("seeds is not set (placeholder)")
    folder = runs.new_run_dir(config_path, cfg)

    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    X, _, ids = transfer.load_set(cfg, meta, cfg["run_id"], cfg["split"])  # labels are not used
    rows = diagnostic_rows(len(X), cfg["diagnostics"]["windows"], cfg["diagnostics"]["seed"])
    windows = pd.concat([pd.DataFrame({"split": cfg["split"], "set": name, "row": np.arange(len(r)), "run_id": cfg["run_id"],
                                       "capture_id": ids[r], "channels": ";".join(cfg["channels"]), "start": 0, "stop": X.shape[1]})
                         for name, r in (("pretrain", np.arange(len(X))), ("diagnostics", rows))], ignore_index=True)
    windows.to_csv(os.path.join(folder, "windows.csv"), index=False)  # D33 (no test windows are used)

    for seed in cfg["seeds"]:
        out = os.path.join(folder, f"seed_{seed}")
        os.makedirs(out)
        runs.set_seed(seed)
        print(f"seed {seed}: pretraining on {len(X)} windows, measuring {len(rows)}", flush=True)
        model, losses, monitor = pretrain(X, X[rows], cfg["tfc"], cfg["diagnostics"], cfg["device"])
        losses.to_csv(os.path.join(out, "pretrain_loss.csv"), index=False)
        pd.DataFrame(monitor.rows).to_csv(os.path.join(out, "collapse.csv"), index=False)
        if monitor.spectra:
            pd.concat(monitor.spectra).to_csv(os.path.join(out, "spectra.csv"), index=False)
        tfc.save(model, os.path.join(out, "pretrained.pt"))


if __name__ == "__main__":
    main(sys.argv[1])
