"""TS2Vec with a split representation (split-representation pilot: D182, D183, D185).

A new named method (D19); the official TS2Vec stays unmodified in external/ts2vec.
Reused from the official code: the encoder (TSEncoder inside the TS2Vec class), the
averaged copy used for encoding, the random cropping and the hierarchical contrastive loss.
The training loop is our own, because the official fit() takes no extra loss terms.
It follows the official loop step by step; the additions are marked "ADDED".

Representation (repr_dims = 320) per timestep: z_op = the first z_op_dims values,
z_state = the rest. Heads act on the max-pooled crop representations (D183 item 2), one per
operating-condition factor (D197; CWRU: load_hp; synthetic: speed_rpm, temperature_c):
  op heads   : MLP on z_op predicts the factor                                     (all variants)
  removal    : 'none'      -> variant A
               'adversary' -> variant B: per factor, an MLP on z_state predicts the factor; its
                              gradient into the encoder is reversed (gradient reversal, Ganin et al. 2016)
               'hsic'      -> variant C: per factor, HSIC between pooled z_state and the factor
Losses of the factors are summed (each factor with the weights of D185).
The TS2Vec loss is applied to all dims (D183 item 3).

Interface as harness/methods/ts2vec.py, plus the condition classes of every pretraining window
(scripts/pretrain.py, D198):
    pretrain(X, s, device, conditions) -> model, loss per iteration
    encode(model, X)                    -> one vector per series (all 320 dims; blocks sliced by the study)
    save(model, path)
"""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from harness.methods.external import import_official

official = import_official("ts2vec", "ts2vec", local_names=("ts2vec", "models", "utils"))
TS2Vec = official.TS2Vec
hierarchical_contrastive_loss = official.hierarchical_contrastive_loss
take_per_row = official.take_per_row

REMOVALS = ("none", "adversary", "hsic")
NEEDS_CONDITIONS = True  # scripts/pretrain.py passes the condition classes (D198)


def check_settings(s):
    """Stop before any work if a setting is missing or still a placeholder (no silent defaults)."""
    needed = ["repr_dims", "hidden_dims", "depth", "lr", "batch_size", "max_train_length", "temporal_unit",
              "n_iters", "z_op_dims", "head_hidden", "op_weight", "removal"]
    if s.get("removal") == "adversary":
        needed += ["removal_weight", "ramp_max"]
    if s.get("removal") == "hsic":
        needed += ["removal_weight"]
    missing = [k for k in needed if s.get(k) is None]
    if missing:
        raise ValueError(f"ts2vec_split settings missing or placeholder: {missing}")
    if s["removal"] not in REMOVALS:
        raise ValueError(f"removal must be one of {REMOVALS}")


class GradReverse(torch.autograd.Function):
    """Identity forward; multiplies the gradient by -coeff backward (gradient reversal layer)."""

    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coeff * grad, None


def mlp(d_in, hidden, n_out):
    """One hidden layer (D185 item 5)."""
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def dann_ramp(p, ramp_max):
    """DANN's schedule (Ganin et al. 2016), scaled to ramp_max: 0 at the start, ramp_max at the end.
    p = training progress in [0, 1]."""
    return ramp_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def hsic(z, c):
    """Biased empirical HSIC (Gretton et al. 2005) between pooled z_state (n, d) and integer conditions (n,).

    Kernels (D183 item 5): Gaussian on z with the median pairwise distance as bandwidth
    (computed without gradient), delta kernel on the condition (1 if equal, else 0).
    HSIC = trace(K H L H) / (n - 1)^2, with H the centring matrix.
    """
    n = z.shape[0]
    d2 = torch.cdist(z, z).pow(2)
    with torch.no_grad():
        off_diag = d2[~torch.eye(n, dtype=torch.bool, device=z.device)]
        sigma2 = off_diag.sqrt().median().pow(2).clamp_min(1e-12)
    K = torch.exp(-d2 / (2.0 * sigma2))
    L = (c[:, None] == c[None, :]).float()
    H = torch.eye(n, device=z.device) - 1.0 / n
    return torch.trace(K @ H @ L @ H) / (n - 1) ** 2


def pretrain(X, s, device, conditions):
    """Train without fault labels; the operating condition of each window supervises the heads.

    X: array (n_series, n_samples, n_channels), pretraining inputs only.
    s: the 'ts2vec_split' section of the config; every setting given explicitly.
    conditions: int array (n_series, n_factors), the class of each factor per window (0..K-1);
                a 1-d array is one factor.
    Returns the model and the total loss per iteration. The parts of the loss per iteration
    are kept on the model (model.loss_parts) and saved next to it.
    """
    check_settings(s)
    conditions = np.asarray(conditions).reshape(len(conditions), -1)  # (n, n_factors)
    assert X.ndim == 3 and len(conditions) == len(X)
    # The official fit() first splits series longer than max_train_length and handles missing values.
    # Neither applies here (windows of fixed length, no NaN); checked instead of copied.
    assert X.shape[1] <= s["max_train_length"] and not np.isnan(X).any()

    model = TS2Vec(input_dims=X.shape[2], output_dims=s["repr_dims"], hidden_dims=s["hidden_dims"],
                   depth=s["depth"], device=device, lr=s["lr"], batch_size=s["batch_size"],
                   max_train_length=s["max_train_length"], temporal_unit=s["temporal_unit"])
    k, d = s["z_op_dims"], s["repr_dims"]
    n_classes = [int(conditions[:, f].max()) + 1 for f in range(conditions.shape[1])]
    # ADDED: heads, trained with the encoder by the same optimiser (one optimiser, D183 item 4).
    # Built on a forked random state, so the shuffling and cropping draw the same numbers as the
    # official fit(): with the added losses off, training is exactly the official one (tested).
    with torch.random.fork_rng(devices=[]):
        model.op_heads = nn.ModuleList(mlp(k, s["head_hidden"], n) for n in n_classes).to(device)
        model.adversaries = (nn.ModuleList(mlp(d - k, s["head_hidden"], n) for n in n_classes).to(device)
                             if s["removal"] == "adversary" else None)
    params = list(model._net.parameters()) + list(model.op_heads.parameters())
    if model.adversaries is not None:
        params += list(model.adversaries.parameters())
    optimizer = torch.optim.AdamW(params, lr=s["lr"])  # the official optimiser and learning rate

    data = TensorDataset(torch.from_numpy(X).to(torch.float), torch.from_numpy(np.asarray(conditions)).long())
    loader = DataLoader(data, batch_size=min(s["batch_size"], len(data)), shuffle=True, drop_last=True)

    losses, parts, it = [], [], 0
    while it < s["n_iters"]:
        for x, c in loader:
            if it >= s["n_iters"]:
                break
            x, c = x.to(device), c.to(device)

            # --- official: two overlapping random crops (ts2vec.py, fit) ---
            ts_l = x.size(1)
            crop_l = np.random.randint(low=2 ** (s["temporal_unit"] + 1), high=ts_l + 1)
            crop_left = np.random.randint(ts_l - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
            optimizer.zero_grad()
            out1 = model._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))[:, -crop_l:]
            out2 = model._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))[:, :crop_l]
            loss_ts2vec = hierarchical_contrastive_loss(out1, out2, temporal_unit=s["temporal_unit"])

            # --- ADDED: heads on the max-pooled overlap of each crop; losses averaged over the two crops ---
            pooled = [out.max(dim=1).values for out in (out1, out2)]  # (batch, repr_dims) each
            factors = range(c.shape[1])
            # summed over factors, averaged over the two crops
            loss_op = sum(F.cross_entropy(model.op_heads[f](z[:, :k]), c[:, f]) for z in pooled for f in factors) / 2
            loss_removal = torch.zeros((), device=device)
            if s["removal"] == "adversary":
                # Each adversary learns to predict its factor from z_state (loss weight 1); the encoder
                # receives that gradient reversed, scaled by removal_weight x the DANN ramp.
                coeff = s["removal_weight"] * dann_ramp(it / s["n_iters"], s["ramp_max"])
                loss_removal = sum(F.cross_entropy(model.adversaries[f](GradReverse.apply(z[:, k:], coeff)), c[:, f])
                                   for z in pooled for f in factors) / 2
                loss = loss_ts2vec + s["op_weight"] * loss_op + loss_removal
            elif s["removal"] == "hsic":
                loss_removal = sum(hsic(z[:, k:], c[:, f]) for z in pooled for f in factors) / 2
                loss = loss_ts2vec + s["op_weight"] * loss_op + s["removal_weight"] * loss_removal
            else:
                loss = loss_ts2vec + s["op_weight"] * loss_op

            loss.backward()
            optimizer.step()
            model.net.update_parameters(model._net)  # official: the averaged copy used for encoding
            losses.append(loss.item())
            parts.append({"ts2vec": loss_ts2vec.item(), "op": loss_op.item(), "removal": loss_removal.item()})
            it += 1
    model.loss_parts = parts
    model.save_live_encoder = bool(s.get("save_live_encoder", False))  # D207; absent = off (diagnostic only)
    return model, losses


def save(model, path):
    """Encoder weights as the official save(); ADDED: heads and the loss parts next to them,
    and with save_live_encoder (D207) also the live encoder's weights (model_live)."""
    model.save(path)
    if getattr(model, "save_live_encoder", False):
        torch.save(model._net.state_dict(), path + "_live")
    heads = {"op_heads": model.op_heads.state_dict(),
             "adversaries": None if model.adversaries is None else model.adversaries.state_dict()}
    torch.save(heads, path + "_heads")
    with open(path + "_loss_parts.json", "w", encoding="utf-8") as f:
        json.dump(model.loss_parts, f)


def encode(model, X):
    """One vector per series (all repr_dims): max-pooling over the whole series, as TS2Vec."""
    return model.encode(X.astype(np.float32), encoding_window="full_series")


def encode_live(model, X):
    """Diagnostic (D207): the same encoding with the live encoder (model._net, the weights the losses
    act on) instead of the official weight-averaged copy (model.net, used by encode())."""
    averaged = model.net
    model.net = model._net  # the official encode() uses self.net; swapped for this call only
    try:
        return encode(model, X)
    finally:
        model.net = averaged
