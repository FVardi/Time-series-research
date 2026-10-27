"""PatchTST (Nie, Nguyen, Sinthong & Kalagnanam, ICLR 2023, arXiv 2211.14730), self-supervised
version, official code, wrapped (D10 mode A, D19).

The official code lives unmodified in external/patchtst (git submodule, commit 204c21e). Only its
folder PatchTST_self_supervised is used (the other folder, PatchTST_supervised, has no pretraining).
Its pretraining script patchtst_pretrain.py reads command-line arguments and creates folders as soon
as it is imported, and its data loaders only read forecasting CSV files, so the script is not called.
Instead, pretrain() below repeats what the script does, step by step, with the official pieces:
the PatchTST model, the Learner (training loop, Adam, one-cycle schedule, learning-rate finder) and
its callbacks (RevInCB, PatchMaskCB, SaveModelCB). Nothing of the training itself is rewritten.

What one training step does (official):
    series [batch, time, channels]
    -> RevIN: each series and channel shifted/scaled to mean 0, std 1 (no learned scale; not undone)
    -> non-overlapping patches [batch, n_patches, channels, patch_len]          (create_patch)
    -> per series AND per channel, a random mask_ratio of the patches set to 0 (random_masking)
    -> encoder: every channel on its own, all channels with the same weights ("channel independence")
    -> linear head: each patch vector back to patch_len values
    -> loss = mean squared error over the masked patches only
    -> Adam step; the learning rate follows a one-cycle schedule, stepped after every batch

Not taken from the official code (it is not in it): our data loaders (plain PyTorch DataLoaders,
set up as the official ones), the optional validation split (see `validation_fraction`), and
encode(), which turns the encoder's patch vectors into one vector per series for the harness probes.

The official Learner chooses the device itself (the GPU if there is one), so the config's device
must agree with it; pretrain() stops if it does not.
"""

import contextlib
import json
import tempfile
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from harness.methods.external import import_official

# The official code is the package 'src' inside PatchTST_self_supervised. All modules we use are
# imported in one go, so every official class comes from the same copy of the package (the Learner
# compares callbacks by type).
official = import_official(
    "patchtst/PatchTST_self_supervised", "src", local_names=("src",),
    preload=("src.basics", "src.models.patchTST", "src.models.layers.revin", "src.learner",
             "src.callback.patch_mask", "src.callback.transforms", "src.callback.tracking",
             "src.callback.scheduler"))


# Fix needed only to make the official code run (D19): its OneCycleLR callback passes verbose=False
# to torch.optim.lr_scheduler.OneCycleLR, an argument newer PyTorch versions (here 2.14) no longer
# accept. 'verbose' only printed the learning rate, so it is dropped; the schedule is unchanged.
def _one_cycle_lr(*args, verbose=False, **kwargs):
    return torch.optim.lr_scheduler.OneCycleLR(*args, **kwargs)


official.callback.scheduler.lr_scheduler = types.SimpleNamespace(OneCycleLR=_one_cycle_lr)


def check_settings(s):
    """Stop if any setting is still an open placeholder (null in the YAML)."""
    def walk(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{path}{k}.")
            elif v is None:
                raise ValueError(f"patchtst.{path}{k} is not set (placeholder); see the PatchTST decision list")
    walk(s, "")


def n_patches(length, patch_len, stride):
    """Number of patches, official formula (patchtst_pretrain.py, get_model).

    If the series does not divide evenly, create_patch drops the FIRST samples:
    the patches cover the last patch_len + stride * (n_patches - 1) samples.
    """
    return (max(length, patch_len) - patch_len) // stride + 1


def build_model(n_vars, length, s):
    """The official PatchTST with the pretraining head, as patchtst_pretrain.py's get_model builds it.

    The script leaves several arguments at the class defaults; here every one is set from the config.
    """
    m = s["model"]
    return official.models.patchTST.PatchTST(
        c_in=n_vars,
        target_dim=1,  # forecast length; not used by the pretraining head (the script passes 96)
        patch_len=s["patch_len"], stride=s["stride"], num_patch=n_patches(length, s["patch_len"], s["stride"]),
        n_layers=m["n_layers"], n_heads=m["n_heads"], d_model=m["d_model"], d_ff=m["d_ff"],
        shared_embedding=m["shared_embedding"], norm=m["norm"], attn_dropout=m["attn_dropout"],
        dropout=m["dropout"], act=m["act"], res_attention=m["res_attention"], pre_norm=m["pre_norm"],
        store_attn=False,  # only keeps attention maps for inspection; no effect on results
        pe=m["pe"], learn_pe=m["learn_pe"], head_dropout=m["head_dropout"], head_type="pretrain")


class Loaders:
    """What the official Learner needs from its data: .train, .valid (None = no validation) and .vars
    (number of channels). Batches are (input, target) pairs as in the official loaders; PatchMaskCB
    replaces the target by the unmasked patches, so the second element is never used."""

    def __init__(self, X_train, X_valid, batch_size):
        self.vars = X_train.shape[2]
        # Official DataLoaders: training data shuffled, validation not; no workers, last batch kept.
        self.train = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True)
        self.valid = None if X_valid is None else DataLoader(TensorDataset(X_valid, X_valid),
                                                             batch_size=batch_size, shuffle=False)


def callbacks(s, n_vars, save_dir=None):
    """The official callback list of patchtst_pretrain.py (same order)."""
    cbs = [official.callback.transforms.RevInCB(n_vars, eps=s["revin_eps"], affine=False, denorm=False)] if s["revin"] else []
    cbs += [official.callback.patch_mask.PatchMaskCB(patch_len=s["patch_len"], stride=s["stride"],
                                                    mask_ratio=s["mask_ratio"])]
    if save_dir is not None:  # keeps the epoch with the lowest validation loss and loads it at the end
        cbs += [official.callback.tracking.SaveModelCB(monitor="valid_loss", fname="best", path=save_dir)]
    return cbs


def pretrain(X, s, device, train=True):
    """Pretrain PatchTST on unlabelled training series X (series, time, channels).

    s: the 'patchtst' section of the config; every setting is given explicitly.
    train: False returns the model as built, untrained (random-init baseline, D134). The validation
           split and the learning-rate finder still run first, so the random numbers drawn before the
           model is built, and hence its initial weights, are the same as in the trained run.
    Returns the model (encoder + pretraining head) and the training loss per epoch.
    """
    check_settings(s)
    used = official.basics.default_device(use_cuda=True).type
    if used != device:
        raise ValueError(f"the official Learner will train on '{used}', but the config says device: {device}")
    length = X.shape[1]
    n = n_patches(length, s["patch_len"], s["stride"])
    print(f"  {n} patches of {s['patch_len']} samples (stride {s['stride']}); "
          f"first {length - (s['patch_len'] + s['stride'] * (n - 1))} samples of each series unused", flush=True)

    # Optional validation split of the training series (not in the official code: its forecasting
    # datasets come with a validation period). Random, without labels; the test set is never used.
    X = torch.from_numpy(X.astype(np.float32))  # D23: float32 at training
    order = torch.randperm(len(X))
    n_valid = int(round(s["validation_fraction"] * len(X)))
    valid_idx, train_idx = order[:n_valid].sort().values, order[n_valid:].sort().values
    dls = Loaders(X[train_idx], X[valid_idx] if n_valid else None, s["batch_size"])

    # Inside a temporary folder: the learning-rate finder saves and reloads the model in 'temp/'
    # under the current folder, and SaveModelCB writes the best model to disk.
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        lr = s["lr"]
        if lr == "finder":
            # Official (find_lr): a first model, used only to suggest a learning rate. The finder raises
            # the learning rate exponentially from 1e-7 to 10 over 100 batches (official defaults of
            # lr_finder) and suggests one from the longest falling stretch of the loss ('valley').
            finder_model = build_model(dls.vars, length, s)
            learn = official.learner.Learner(dls, finder_model, torch.nn.MSELoss(reduction="mean"),
                                             lr=1e-4,  # script's --lr default; replaced by the finder's 1e-7 before any step
                                             cbs=callbacks(s, dls.vars))
            lr = float(learn.lr_finder(show_plot=False))  # show_plot only draws a figure
            print(f"  learning-rate finder suggests {lr:.3g}", flush=True)
        elif not isinstance(lr, (int, float)):
            raise ValueError(f"patchtst.lr must be a number or 'finder', not {lr!r}")

        # Official (pretrain_func): a NEW model, trained with fit_one_cycle. The loss given here is
        # replaced by PatchMaskCB's masked MSE. Adam (PyTorch defaults otherwise); one-cycle schedule:
        # the learning rate rises from lr/25 to lr over the first pct_start of the batches, then falls
        # (cosine) to lr/25/1e4; Adam's beta1 moves the other way between 0.95 and 0.85 (official
        # OneCycleLR callback, fixed in the official code).
        model = build_model(dls.vars, length, s)
        if train:
            learn = official.learner.Learner(dls, model, torch.nn.MSELoss(reduction="mean"), lr=lr,
                                             cbs=callbacks(s, dls.vars, save_dir=tmp if n_valid else None))
            learn.fit_one_cycle(n_epochs=s["epochs"], lr_max=lr, pct_start=s["pct_start"])
            rec = learn.recorder
        else:  # random-init baseline (D134): the model as built above, no training
            model = model.to(device)
            rec = {"train_loss": []}

    # Without validation the last epoch is kept; with validation SaveModelCB has loaded the best one.
    valid_loss = [float(v) for v in rec.get("valid_loss", [])]
    model.training_record = {
        "lr": lr,
        "train_loss": [float(v) for v in rec["train_loss"]],
        "valid_loss": valid_loss,
        "epoch_kept": (int(np.argmin(valid_loss)) if valid_loss else s["epochs"] - 1) if train else None,
        "validation_rows": valid_idx.tolist(),  # rows of the training series held out for model selection
    }
    model.encode_settings = encode_settings(s)
    return model, model.training_record["train_loss"]


def encode_settings(s):
    """Settings encode() needs, kept with the model (the harness calls encode(model, X))."""
    return {k: s[k] for k in ("patch_len", "stride", "revin", "revin_eps", "pooling", "encode_batch_size")}


def load(path, n_vars, length, s, device):
    """A model saved by save() back, ready for encode() with the settings s (which may set another pooling).

    n_vars, length: number of channels and series length the model was trained on (they fix its shape).
    """
    model = build_model(n_vars, length, s)
    model.load_state_dict(torch.load(path + ".pth", map_location=device))
    model.encode_settings = encode_settings(s)
    return model.to(device)


@torch.no_grad()
def encode(model, X):
    """One vector per series (and channel): RevIN and patching as in training, no masking, then the
    encoder's patch vectors [channels, d_model, n_patches] pooled over patches, channels concatenated.

    Pooling (see `pooling` in the config):
      last    - the last patch's vector: what the official ClassificationHead and RegressionHead use
      mean    - mean over the patch vectors
      flatten - all patch vectors (what the official forecasting head uses; d_model * n_patches per channel)
    Channels are concatenated as in all official heads.
    """
    model.eval()
    e = model.encode_settings
    device = next(model.parameters()).device
    revin = official.models.layers.revin.RevIN(X.shape[2], eps=e["revin_eps"], affine=False)
    out = []
    for i in range(0, len(X), e["encode_batch_size"]):
        x = torch.from_numpy(X[i:i + e["encode_batch_size"]].astype(np.float32)).to(device)
        if e["revin"]:
            x = revin(x, "norm")  # statistics of each series and channel separately (over time)
        patches, _ = official.callback.patch_mask.create_patch(x, e["patch_len"], e["stride"])
        z = model.backbone(patches)  # [batch, channels, d_model, n_patches]
        if e["pooling"] == "last":
            z = z[:, :, :, -1]
        elif e["pooling"] == "mean":
            z = z.mean(dim=3)
        elif e["pooling"] == "flatten":
            z = z.flatten(2)
        else:
            raise ValueError(f"patchtst.pooling must be 'last', 'mean' or 'flatten', not {e['pooling']!r}")
        out.append(z.flatten(1).cpu())  # channels concatenated
    return torch.cat(out).numpy()


def save(model, path):
    """Save the weights in the official checkpoint format (plain state_dict of the whole model, as
    Learner.save writes it; the official fine-tuning script loads it with transfer_weights), and the
    training record (learning rate used, losses, epoch kept, validation rows) next to it."""
    torch.save(model.state_dict(), path + ".pth")
    with open(path + "_training.json", "w") as f:
        json.dump(model.training_record, f, indent=1)
