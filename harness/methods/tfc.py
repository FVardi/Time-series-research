"""TF-C, our implementation (D35 step 3, D36, D41).

Follows the TF-C paper (Zhang et al., 2022, arXiv 2206.08496) and fixes the problems
found in the official code (decisions/training.md, D36 and D41). In short:

    x  ──► time branch:      G_T (CNN) ─► h_T ─► R_T (projector) ─► z_T
    |FFT(x)| ─► frequency branch: G_F (CNN) ─► h_F ─► R_F (projector) ─► z_F

Pretraining loss (paper, eq. 1-3):  λ (L_T + L_F) + (1 − λ) L_C
    L_T = NT-Xent(h_T, h̃_T)      time view vs its augmented version
    L_F = NT-Xent(h_F, h̃_F)      frequency view vs its augmented version
    L_C = Σ_pairs (S_TF − S_pair + δ), S = NT-Xent between projected embeddings,
          pairs: (z_T, z̃_F), (z̃_T, z_F), (z̃_T, z̃_F)
Fine-tuning: cross-entropy of a small classifier on [z_T; z_F] + the same loss,
encoder and classifier trained together. Representation for probes: [z_T; z_F].

Every setting comes from the config (section 'tfc'); nothing has a default here.
Inputs are arrays (series, time, channels) float64; they are converted to 32-bit
floats on the device (D23).
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


# ----------------------------------------------------------------------------- settings

def check_settings(s):
    """Stop if any setting is still an open placeholder (null in the YAML)."""
    def walk(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{path}{k}.")
            elif v is None:
                raise ValueError(f"tfc.{path}{k} is not set (placeholder); see D41")
    walk(s, "")


# ----------------------------------------------------------------------------- model

def conv_block(c_in, c_out, e, stride, dropout):
    """Conv1d → BatchNorm → ReLU → MaxPool(2), optionally followed by dropout (as the 2022 code)."""
    layers = [nn.Conv1d(c_in, c_out, kernel_size=e["kernel_size"], stride=stride,
                        padding=e["padding"], bias=e["conv_bias"]),
              nn.BatchNorm1d(c_out),
              nn.ReLU(),
              nn.MaxPool1d(kernel_size=2, stride=2, padding=e["pool_padding"])]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Branch(nn.Module):
    """Encoder G (3 conv blocks, D41 item 1) and projector R for one domain (time or frequency)."""

    def __init__(self, length, in_channels, e):
        super().__init__()
        c = e["channels"]  # [32, 64, 128]
        st = e["strides"]  # [8, 1, 1]
        self.encoder = nn.Sequential(
            conv_block(in_channels, c[0], e, st[0], e["dropout_block1"]),
            conv_block(c[0], c[1], e, st[1], 0),
            conv_block(c[1], c[2], e, st[2], 0),
        )
        with torch.no_grad():  # size of the flattened encoder output h
            n_h = self.encoder(torch.zeros(1, in_channels, length)).numel()
        hidden, out = e["projector"]  # [256, 128]
        self.projector = nn.Sequential(nn.Linear(n_h, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                       nn.Linear(hidden, out))

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        return h, self.projector(h)


class TFC(nn.Module):
    """Time branch and frequency branch, with separate parameters (paper)."""

    def __init__(self, length, in_channels, e):
        super().__init__()
        self.time = Branch(length, in_channels, e)
        self.freq = Branch(length, in_channels, e)

    def forward(self, x_t, x_f):
        h_t, z_t = self.time(x_t)
        h_f, z_f = self.freq(x_f)
        return h_t, z_t, h_f, z_f


class Classifier(nn.Module):
    """Fine-tuning head on [z_T; z_F]: Linear → sigmoid → Linear (2-layer head, hidden 64; sigmoid as the code)."""

    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.hidden = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, z):
        return self.out(torch.sigmoid(self.hidden(z)))


# ----------------------------------------------------------------------------- views

def spectrum(x):
    """Full FFT magnitude spectrum, same length as x (D41 item 2). Symmetric: |X_k| = |X_(N-k)|."""
    return torch.fft.fft(x, dim=-1).abs()


def freq_augment(xf, n_components, alpha):
    """Remove and add frequency components in the same spectrum, symmetrically (D41 item 3).

    xf: (batch, channels, N) magnitude spectra, N even. For each window and channel:
      - removal: n_components random bins k get amplitude 0;
      - addition: n_components other bins, among those with amplitude < alpha * max,
        get amplitude alpha * max (max = the largest amplitude of that spectrum);
      - every change is applied to bin k and its mirror N - k, so the result is still
        the magnitude spectrum of a real signal.
    Only bins 1 .. N/2 - 1 are changed: the constant (k = 0) and the Nyquist bin
    (k = N/2) have no mirror partner. A removed bin is never chosen for addition.
    """
    B, C, N = xf.shape
    assert N % 2 == 0, "spectrum length must be even"
    half = N // 2
    amp = xf[..., 1:half].reshape(B * C, half - 1)  # one row per spectrum, bins 1 .. N/2 - 1
    top = alpha * xf.reshape(B * C, N).amax(dim=1, keepdim=True)  # alpha * max, per spectrum

    # Removal: n random bins per spectrum.
    remove = torch.rand(amp.shape, device=xf.device).topk(n_components, dim=1).indices
    removed = torch.zeros_like(amp, dtype=torch.bool).scatter(1, remove, True)

    # Addition: n random bins among the weak ones that were not removed.
    can_add = (amp < top) & ~removed
    score = torch.rand(amp.shape, device=xf.device).masked_fill(~can_add, -1.0)
    best, add = score.topk(n_components, dim=1)
    ok = best >= 0  # False only if fewer than n weak bins exist; then that addition is skipped

    new = amp.scatter(1, remove, 0.0)
    current = new.gather(1, add)
    new = new.scatter(1, add, torch.where(ok, top.expand_as(current), current))

    out = xf.clone().reshape(B * C, N)
    out[:, 1:half] = new
    out[:, half + 1:] = new.flip(1)  # mirror: bin N - k gets the value of bin k
    return out.reshape(B, C, N)


def jitter(x, sigma):
    """Add Gaussian noise with standard deviation sigma."""
    return x + sigma * torch.randn_like(x)


def scaling(x, mean, sigma):
    """Multiply each window and channel by one factor drawn from N(mean, sigma)."""
    return x * (mean + sigma * torch.randn(x.shape[0], x.shape[1], 1, device=x.device))


def permutation(x, max_segments):
    """Permutation (D42), as described by TS-TCC: "splitting the signal into a random number
    of segments with a maximum of M and randomly shuffling them".

    Number of segments: uniform in 1 .. M (the text's maximum M; the official code's
    randint(1, M) only reaches M - 1). Cut points: drawn as in the official code,
    np.random.choice(L - 2, n - 1, replace=False), sorted. One segment = unchanged window.
    """
    B, _, L = x.shape
    idx = np.empty((B, L), dtype=np.int64)
    n_segs = np.random.randint(1, max_segments + 1, size=B)
    for i in range(B):
        order = np.arange(L)
        if n_segs[i] > 1:
            cuts = np.sort(np.random.choice(L - 2, n_segs[i] - 1, replace=False))
            pieces = np.split(order, cuts)
            order = np.concatenate([pieces[j] for j in np.random.permutation(len(pieces))])
        idx[i] = order
    idx = torch.from_numpy(idx).to(x.device)
    return x.gather(2, idx[:, None, :].expand_as(x))


def time_augment(x, a):
    """One of the three time augmentations, chosen at random per window (paper: one augmented
    sample is selected per window; set of augmentations: D42)."""
    options = torch.stack([jitter(x, a["jitter_sigma"]),
                           scaling(x, a["scaling_mean"], a["scaling_sigma"]),
                           permutation(x, a["max_segments"])])  # (3, B, C, L)
    choice = torch.randint(0, len(options), (x.shape[0],), device=x.device)
    return options[choice, torch.arange(x.shape[0], device=x.device)]


# ----------------------------------------------------------------------------- losses

def nt_xent(a, b, tau):
    """NT-Xent (SimCLR) between two batches; pairs (a_i, b_i) are positives,
    all other 2B - 2 embeddings in the two batches are negatives. Cosine similarity.
    Same value as the official NTXentLoss (tested in tests/test_tfc.py)."""
    B = a.shape[0]
    z = F.normalize(torch.cat([a, b]), dim=1)
    sim = z @ z.T / tau
    sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))  # no self-pairs
    positive = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, positive)  # mean over the 2B rows


def tfc_loss(out, out_aug, l):
    """Pretraining loss λ(L_T + L_F) + (1 − λ)L_C (paper, eq. 1-3; D41 item 4).

    out, out_aug: (h_t, z_t, h_f, z_f) for the original and the augmented views.
    """
    h_t, z_t, h_f, z_f = out
    h_ta, z_ta, h_fa, z_fa = out_aug
    tau, lam, delta = l["temperature"], l["lambda"], l["delta"]
    loss_t = nt_xent(h_t, h_ta, tau)
    loss_f = nt_xent(h_f, h_fa, tau)
    s_tf = nt_xent(z_t, z_f, tau)
    loss_c = sum(s_tf - nt_xent(p, q, tau) + delta for p, q in ((z_t, z_fa), (z_ta, z_f), (z_ta, z_fa)))
    return lam * (loss_t + loss_f) + (1 - lam) * loss_c


# ----------------------------------------------------------------------------- training

def to_device(X, device):
    """(series, time, channels) float64 -> (series, channels, time) float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(np.transpose(X, (0, 2, 1)))).float().to(device)


def batches(n, batch_size, drop_last):
    """Random batches of indices for one epoch."""
    perm = torch.randperm(n)
    stop = n - n % batch_size if drop_last else n
    return [perm[i:i + batch_size] for i in range(0, stop, batch_size)]


def adam(params, o):
    return torch.optim.Adam(params, lr=o["lr"], betas=tuple(o["betas"]), weight_decay=o["weight_decay"])


def augmented_step(model, x, s):
    """Forward pass of a batch and of its augmented views; returns both outputs."""
    xf = spectrum(x)
    out = model(x, xf)
    out_aug = model(time_augment(x, s["time_aug"]), freq_augment(xf, s["freq_aug"]["n_components"],
                                                                 s["freq_aug"]["alpha"]))
    return out, out_aug


def pretrain(X, s, device):
    """Pretrain on unlabelled windows X (series, time, channels). Returns the model and the loss per epoch."""
    check_settings(s)
    x_all = to_device(X, device)
    model = TFC(x_all.shape[2], x_all.shape[1], s["encoder"]).to(device)
    p = s["pretrain"]
    opt = adam(model.parameters(), p)
    log = []
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        losses = []
        for idx in batches(len(x_all), p["batch_size"], p["drop_last"]):
            loss = tfc_loss(*augmented_step(model, x_all[idx.to(device)], s), s["loss"])
            opt.zero_grad()  # every step (the official code resets once per epoch)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"  pretrain epoch {epoch}: loss {log[-1]['loss']:.4f}", flush=True)
    return model, pd.DataFrame(log)


@torch.no_grad()
def predict(model, classifier, x_all, chunk):
    """Class probabilities, in evaluation mode, in chunks (results do not depend on the chunk)."""
    model.eval()
    classifier.eval()
    probs = []
    for i in range(0, len(x_all), chunk):
        x = x_all[i:i + chunk]
        _, z_t, _, z_f = model(x, spectrum(x))
        probs.append(F.softmax(classifier(torch.cat([z_t, z_f], dim=1)), dim=1).cpu())
    return torch.cat(probs).numpy()


def scores(y, probs):
    """Metrics on a whole set (D41 item 8). Macro averages; zero_division=0 for empty classes."""
    pred = probs.argmax(1)
    onehot = np.eye(probs.shape[1])[y]
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, average="macro", zero_division=0),
            "recall": recall_score(y, pred, average="macro", zero_division=0),
            "f1": f1_score(y, pred, average="macro", zero_division=0),
            "auroc": roc_auc_score(y, probs, average="macro", multi_class="ovr"),
            "auprc": average_precision_score(onehot, probs, average="macro")}


def finetune(model, data, s, device):
    """Fine-tune encoder + classifier on the target training set (D41 items 6-7).

    data: dict with 'target_train', 'target_val', 'target_test', each (X, labels 0..K-1).
    The epoch with the best validation macro-F1 is kept (first one if tied); the test
    set is scored once, for that epoch. Returns (per-epoch table, test scores, best epoch).
    """
    check_settings(s)
    f = s["finetune"]
    model = copy.deepcopy(model)  # the pretrained model stays as it was
    x_tr, y_tr = to_device(data["target_train"][0], device), torch.tensor(np.asarray(data["target_train"][1]), dtype=torch.long, device=device)
    x_val, y_val = to_device(data["target_val"][0], device), data["target_val"][1]
    n_classes = int(y_tr.max().item()) + 1
    proj = s["encoder"]["projector"][1]
    classifier = Classifier(2 * proj, s["classifier"]["hidden"], n_classes).to(device)
    opt_model, opt_clf = adam(model.parameters(), f), adam(classifier.parameters(), f)

    rows, best = [], None
    for epoch in range(1, f["epochs"] + 1):
        model.train()
        classifier.train()
        losses = []
        for idx in batches(len(x_tr), f["batch_size"], f["drop_last"]):
            idx = idx.to(device)
            out, out_aug = augmented_step(model, x_tr[idx], s)
            _, z_t, _, z_f = out
            loss = F.cross_entropy(classifier(torch.cat([z_t, z_f], dim=1)), y_tr[idx]) + tfc_loss(out, out_aug, s["loss"])
            opt_model.zero_grad()
            opt_clf.zero_grad()
            loss.backward()
            opt_model.step()
            opt_clf.step()
            losses.append(loss.item())

        val = scores(y_val, predict(model, classifier, x_val, s["eval_chunk"]))
        rows.append({"epoch": epoch, "finetune_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}})
        if best is None or val["f1"] > best[0]:  # strictly better: the first best epoch is kept
            best = (val["f1"], epoch, copy.deepcopy(model.state_dict()), copy.deepcopy(classifier.state_dict()))
        print(f"  finetune epoch {epoch}: loss {rows[-1]['finetune_loss']:.4f}, val F1 {val['f1']:.4f}", flush=True)

    _, best_epoch, model_state, clf_state = best
    model.load_state_dict(model_state)
    classifier.load_state_dict(clf_state)
    x_te, y_te = to_device(data["target_test"][0], device), data["target_test"][1]
    test = scores(y_te, predict(model, classifier, x_te, s["eval_chunk"]))  # the only use of the test set
    return pd.DataFrame(rows), test, best_epoch


@torch.no_grad()
def encode(model, X, chunk, device):
    """Representation [z_T; z_F] per window (paper), for the harness probes (D41 item 10)."""
    model.eval()
    x_all = to_device(X, device)
    out = []
    for i in range(0, len(x_all), chunk):
        x = x_all[i:i + chunk]
        _, z_t, _, z_f = model(x, spectrum(x))
        out.append(torch.cat([z_t, z_f], dim=1).cpu())
    return torch.cat(out).numpy()


def save(model, path):
    """Save the weights (about 22 MB for FD windows of 5,120 samples)."""
    torch.save(model.state_dict(), path)
