"""TS-JEPA (Ennadir, Golkar & Sarra, 2025, arXiv 2509.25449), official code, wrapped (D10 mode A, D19).

The official code lives unmodified in external/tsjepa (git submodule, commit 830ee5c).
Its pretrain.py is a script (training loop under `if __name__ == "__main__"`) with a data
loader that only reads one CSV file, so neither is called. Instead, the training loop of
pretrain.py is written out below; everything inside it comes from the official code:
Encoder, Predictor, apply_mask, init_weights and loss_pred.

What one training step does (official):
    series -> n_patches non-overlapping patches
    per series: a random set of patches is masked (mask_ratio), the rest is the context
    target  = layer_norm(EMA encoder(all patches)), taken at the masked positions
    context = encoder(context patches only)
    pred    = predictor(context, positions of the masked patches)
    loss    = mean |pred - target|          (L1)
    then: optimiser step on encoder + predictor; EMA encoder <- m * EMA + (1 - m) * encoder

Not taken from the official code (they are not in it): splitting our series into patches
(the official loader hard-codes 10 patches of 32 samples from one CSV column), and encode(),
which turns the encoder's output per patch into one vector per series.

Official-code bug (see `masking` in the config): apply_mask expects a list of [batch, K]
index tensors (as in I-JEPA) but pretrain.py passes one [batch, K] tensor. It then loops over
the batch rows and takes every series' tokens from the FIRST series of the batch. Setting
masking: official keeps this behaviour; masking: per_sample passes the tensor inside a list, so
each series uses its own tokens. The official files are not edited in either case.
"""

import copy
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from harness.methods.external import import_official

# pretrain.py imports Encoder, Predictor, apply_mask and init_weights itself, so one import
# gives all official pieces from the same files. Importing it runs warnings.filterwarnings("ignore")
# (switches off all warnings in the process); the filters are restored right after.
_filters = warnings.filters[:]
official = import_official("tsjepa", "pretrain", local_names=("pretrain", "src", "main", "config"))
warnings.filters[:] = _filters


def check_settings(s):
    """Stop if any setting is still an open placeholder (null in the YAML)."""
    def walk(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{path}{k}.")
            elif v is None:
                raise ValueError(f"tsjepa.{path}{k} is not set (placeholder); see the TS-JEPA decision list")
    walk(s, "")


def to_patches(X, n_patches):
    """(series, time, 1) float64 -> (series, n_patches, patch_length) float32 tensor (D23).

    Consecutive, non-overlapping patches (paper). The official tokenizer takes one channel only.
    """
    if X.shape[2] != 1:
        raise ValueError("TS-JEPA's tokenizer takes one channel (Conv1d with in_channels=1)")
    n, length, _ = X.shape
    if length % n_patches:
        raise ValueError(f"series length {length} is not divisible into {n_patches} patches")
    return torch.from_numpy(X[:, :, 0].reshape(n, n_patches, length // n_patches).astype(np.float32))


def random_masks(batch_size, n_patches, n_masked):
    """As the official data loader, per series: n_masked patch indices drawn with random.sample
    (in the order drawn), the remaining indices in ascending order. Returns two [batch, K] tensors."""
    masked, context = [], []
    for _ in range(batch_size):
        m = random.sample(range(n_patches), n_masked)
        masked.append(m)
        context.append([i for i in range(n_patches) if i not in m])
    return torch.tensor(masked), torch.tensor(context)


def shuffled(n):
    """A random order of 0..n-1, drawn exactly as torch's DataLoader(shuffle=True) draws it,
    so the batches are the same as in the official training loop: the DataLoader first draws a
    seed for worker processes (unused without workers), then RandomSampler draws its own seed."""
    torch.empty((), dtype=torch.int64).random_()  # DataLoader's worker base seed (discarded)
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def pretrain(X, s, device):
    """Pretrain TS-JEPA on unlabelled training series X (series, time, 1).

    s: the 'tsjepa' section of the config; every setting is given explicitly.
    Returns the online encoder (the one the official code saves and evaluates) and the loss per epoch.
    """
    check_settings(s)
    P = to_patches(X, s["n_patches"])
    n_patches, patch_length = P.shape[1], P.shape[2]
    e, p, o = s["encoder"], s["predictor"], s["optimizer"]

    encoder = official.Encoder(num_patches=n_patches, dim_in=patch_length, kernel_size=e["kernel_size"],
                               embed_dim=e["embed_dim"], embed_bias=e["embed_bias"], nhead=e["nhead"],
                               num_layers=e["num_layers"], jepa=True)
    predictor = official.Predictor(num_patches=n_patches, encoder_embed_dim=e["embed_dim"],
                                   predictor_embed_dim=p["embed_dim"], nhead=p["nhead"], num_layers=p["num_layers"])
    for m in list(encoder.modules()) + list(predictor.modules()):
        official.init_weights(m)  # official: truncated normal (std 0.02) for Linear layers, LayerNorm 1/0

    # Official: AdamW over encoder + predictor, learning rate decaying linearly to lr_end_factor * lr
    # over the epochs. AdamW's betas/eps/weight decay are PyTorch defaults in the official code; here explicit.
    optimizer = torch.optim.AdamW([{"params": encoder.parameters()}, {"params": predictor.parameters()}],
                                  lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"], weight_decay=o["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=o["lr_end_factor"],
                                                  total_iters=s["epochs"])
    encoder, predictor = encoder.to(device), predictor.to(device)
    encoder_ema = copy.deepcopy(encoder)
    for q in encoder_ema.parameters():
        q.requires_grad = False

    n_masked = int(n_patches * s["mask_ratio"])  # official: int(num_patches * mask_ratio)
    if s["masking"] == "official":
        wrap = lambda idx: idx  # one [batch, K] tensor: tokens taken from the first series (official bug)
    elif s["masking"] == "per_sample":
        wrap = lambda idx: [idx]  # a list of one [batch, K] tensor: each series its own tokens
    else:
        raise ValueError(f"tsjepa.masking must be 'official' or 'per_sample', not {s['masking']!r}")

    log = []
    for epoch in range(s["epochs"]):
        scheduler.step()  # official order: stepped at the start of each epoch, before any optimiser step
        # EMA momentum, official schedule (per epoch): m0 + epoch * (1 - m0) / (epochs * ipe_scale)
        m = s["ema_momentum"] + epoch * (1 - s["ema_momentum"]) / (s["epochs"] * s["ipe_scale"])
        encoder.train()
        predictor.train()
        losses = []
        for idx in shuffled(len(P)).split(s["batch_size"]):  # official: DataLoader(shuffle=True)
            patches = P[idx].to(device)
            masked, context = (t.to(device) for t in random_masks(len(idx), n_patches, n_masked))
            optimizer.zero_grad()
            with torch.no_grad():  # targets from the EMA encoder, normalised over the feature dimension
                target = encoder_ema(patches)
                target = F.layer_norm(target, (target.size(-1),))
                target = official.apply_mask(target, wrap(masked))
            tokens = encoder(patches, mask=wrap(context))  # the encoder sees the context patches only
            # The predictor's own apply_mask acts on positional embeddings, identical for every series,
            # so passing the plain tensors there gives the same result with or without the bug.
            pred = predictor(tokens, mask=masked, non_masks=context)
            loss = official.loss_pred(pred, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for q, k in zip(encoder.parameters(), encoder_ema.parameters()):
                    k.data.mul_(m).add_((1.0 - m) * q.detach().data)
            losses.append(loss.item())
        log.append(float(np.mean(losses)))
        if epoch % s["print_every"] == 0:
            print(f"  epoch {epoch}: loss {log[-1]:.4f}", flush=True)

    # Settings encode() needs, kept with the model (the harness calls encode(model, X)).
    encoder.n_patches, encoder.pooling, encoder.encode_batch_size = s["n_patches"], s["pooling"], s["encode_batch_size"]
    return encoder, log


@torch.no_grad()
def encode(model, X):
    """One vector per series: the encoder's output for all patches (no masking), pooled over patches.

    Pooling is not given in the paper; see `pooling` in the config.
    """
    model.eval()
    device = next(model.parameters()).device
    P = to_patches(X, model.n_patches)
    out = []
    for i in range(0, len(P), model.encode_batch_size):
        tokens = model(P[i:i + model.encode_batch_size].to(device))  # [batch, n_patches, embed_dim]
        if model.pooling == "mean":
            z = tokens.mean(dim=1)
        elif model.pooling == "flatten":
            z = tokens.flatten(1)
        else:
            raise ValueError(f"tsjepa.pooling must be 'mean' or 'flatten', not {model.pooling!r}")
        out.append(z.cpu())
    return torch.cat(out).numpy()


def save(model, path):
    """Save the encoder weights in the official checkpoint format ({'encoder': state_dict})."""
    torch.save({"encoder": model.state_dict()}, path + ".pt")
