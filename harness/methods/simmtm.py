"""SimMTM (Dong et al., NeurIPS 2023), official classification code, wrapped (D10 mode A, D19).

The official code lives unmodified in external/simmtm (git submodule, commit 169513b);
the classification part is SimMTM_Classification/code. Its main.py is a script that
reads files from fixed paths and fine-tunes after every second pretraining epoch, so it
is not run. Instead, the steps main.py and Trainer() take for pretraining are written
out below, in the same order, and the official code does everything inside them:
    Load_Dataset   builds the training set (shuffles; keeps channel 0, first TSlength_aligned samples)
    TFC            the model: 3-block CNN encoder (TF-C's time encoder), projector,
                   series-wise contrastive loss, point-wise aggregation, decoder, loss weighting
    model_pretrain one epoch: M masked copies per series, forward, backward, optimiser step,
                   scheduler step
Left out: fine-tuning and testing (the harness probes the frozen encoder instead),
logging and checkpoint files.

What one training step does (official):
    batch (B series) -> M = positive_nums masked copies each (geometric masks, ratio r, mean
    masked length lm; masked values set to 0); encoder on all B(M+1) series
    loss_cl: each series vs its own copies (positives) against all others, temperature τ
    rebuild: each series' point-wise features = softmax(similarity / τ, itself excluded)
             weighted sum of the others' features; decoder -> series; MSE to the inputs
    loss = Kendall-style automatic weighting of loss_cl and loss_rb (learnt weights)

Two fixes needed to run on other series lengths (D19: fixes go in the adapter):
  - the decoder is hard-coded as nn.Linear(1280, 178), the sizes for SleepEEG
    (10 x 128 encoder values -> 178 samples). It is replaced by a layer of the same kind
    with the sizes of our data, only when they differ (so the SleepEEG case stays identical).
  - trainer.py appends '..' to the import path and switches off all warnings when imported;
    both are undone after the import.

    pretrain(X, settings, device) -> (model, loss per epoch)
    encode(model, X)              -> one vector per series
    save(model, path)
"""

import sys
import types
import warnings

import numpy as np
import torch

from harness.methods.external import import_official

CODE = "simmtm/SimMTM_Classification/code"
NAMES = ("model", "loss", "trainer", "utils", "layers", "dataloader", "config_files")
_filters, _path = warnings.filters[:], sys.path[:]
official_trainer = import_official(CODE, "trainer", local_names=NAMES)  # also brings model.py and loss.py
official_dataloader = import_official(CODE, "dataloader", local_names=NAMES)
SleepEEGConfig = import_official(CODE, "config_files.SleepEEG_Configs", local_names=NAMES).Config
warnings.filters[:], sys.path[:] = _filters, _path  # undo trainer.py's side effects


def check_settings(s):
    """Stop if any setting is still an open placeholder (null in the YAML)."""
    def walk(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{path}{k}.")
            elif v is None:
                raise ValueError(f"simmtm.{path}{k} is not set (placeholder); see the SimMTM decision list")
    walk(s, "")


def _attribute(configs, name):
    """'Context_Cont.temperature' -> (configs.Context_Cont, 'temperature'). Stops if it does not exist."""
    obj = configs
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise ValueError(f"{name}: not in the official SleepEEG config")
    return obj, parts[-1]


def official_config(s):
    """The official SleepEEG config (the only pretraining config in the repository) with
    the values s['changed'] changes for our data.

    s['official_config']: every SleepEEG value, as written in the official file;
                          a difference stops the run, so nothing is changed silently.
    """
    configs = SleepEEGConfig()
    for name, value in s["official_config"].items():
        obj, attr = _attribute(configs, name)
        if getattr(obj, attr) != value:
            raise ValueError(f"{name}: YAML says {value}, official config says {getattr(obj, attr)}")
    for name, value in s["changed"].items():
        obj, attr = _attribute(configs, name)
        setattr(obj, attr, value)
    return configs


def main_args(s):
    """The command-line arguments main.py passes on to the model and model_pretrain."""
    a = s["main_args"]
    return types.SimpleNamespace(training_mode="pre_train", temperature=a["temperature"],
                                 positive_nums=a["positive_nums"], masking_ratio=a["masking_ratio"], lm=a["lm"])


def encoder_length(model, configs):
    """Length of the encoder output for a series of TSlength_aligned samples."""
    was_training = model.training
    model.eval()  # no dropout, no update of the batch-norm statistics
    with torch.no_grad():
        x = torch.zeros(1, configs.input_channels, configs.TSlength_aligned, device=next(model.parameters()).device)
        length = model.conv_block3(model.conv_block2(model.conv_block1(x))).shape[2]
    model.train(was_training)
    return length


def build_model(configs, args, device):
    """TFC as main.py builds it, checked against the data sizes, decoder sized for our data."""
    model = official_trainer.TFC(configs, args).to(device)
    length = encoder_length(model, configs)
    if length != configs.CNNoutput_channel:  # sizes the projector's first layer (dense)
        raise ValueError(f"CNNoutput_channel is {configs.CNNoutput_channel}, the encoder output has length {length}")
    n_in, n_out = configs.CNNoutput_channel * configs.final_out_channels, configs.TSlength_aligned
    if (model.head.in_features, model.head.out_features) != (n_in, n_out):  # official: nn.Linear(1280, 178)
        model.head = torch.nn.Linear(n_in, n_out).to(device)
    return model


def pretrain(X, s, device, train=True):
    """Self-supervised pretraining as main.py and Trainer() do it.

    X: array (n_series, n_samples, 1), training inputs only.
    s: the 'simmtm' section of the config; every setting is given explicitly.
    train: False returns the model as built, untrained (random-init baseline, D134).
    The seeds are set before this call (scripts/pretrain.py); main.py's cuDNN settings are repeated here.
    Returns the model and the mean training loss per epoch.
    """
    check_settings(s)
    configs, args = official_config(s), main_args(s)
    if X.shape[2] != configs.input_channels or X.shape[1] != configs.TSlength_aligned:
        raise ValueError(f"the data have {X.shape[2]} channel(s) and {X.shape[1]} samples; the config says "
                         f"{configs.input_channels} and {configs.TSlength_aligned} (Load_Dataset would cut silently)")
    torch.backends.cudnn.deterministic = False  # as set_seed() in main.py
    torch.backends.cudnn.benchmark = False

    # The official data_generator(), training part. Labels are not used; zeros fill their slot.
    samples = torch.from_numpy(np.transpose(X, (0, 2, 1)).astype(np.float32))  # (n, channels, time), 32-bit (D23)
    data = {"samples": samples, "labels": torch.zeros(len(X), dtype=torch.long)}
    dataset = official_dataloader.Load_Dataset(data, configs, "pre_train", subset=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True,
                                         drop_last=configs.drop_last, num_workers=0)

    # Model, optimiser and schedule as in main.py (weight decay 0 hard-coded there; eps not set = PyTorch's).
    model = build_model(configs, args, device)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=s["main_args"]["pretrain_lr"],
                                 betas=(configs.beta1, configs.beta2), eps=s["adam"]["eps"],
                                 weight_decay=s["adam"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=s["main_args"]["pretrain_epoch"])
    model.encode_batch_size = s["encode_batch_size"]  # kept with the model for encode()
    if not train:  # random-init baseline (D134): the model as built above, no training
        return model, []

    log = []
    for epoch in range(1, s["main_args"]["pretrain_epoch"] + 1):
        loss, loss_cl, loss_rb = official_trainer.model_pretrain(model, optimizer, scheduler, loader, configs, args, device)
        log.append(float(loss))
        print(f"  epoch {epoch}: loss {float(loss):.4f} (contrastive {float(loss_cl):.4f}, "
              f"reconstruction {float(loss_rb):.4f})", flush=True)
    return model, log


@torch.no_grad()
def encode(model, X):
    """One vector per series: h, the flattened encoder output, which the official fine-tuning
    feeds to its classifier (model(data) returns h, z; fea_concat = h). Evaluation mode, so the
    batch size only affects speed, not the values."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(np.transpose(X, (0, 2, 1)).astype(np.float32))
    out = []
    for batch in torch.split(x, model.encode_batch_size):
        h, _ = model(batch.to(device))
        out.append(h.cpu())
    return torch.cat(out).numpy()


def save(model, path):
    """Save the weights in the official checkpoint format (as ckp_ep<k>.pt in the official code)."""
    torch.save({"model_state_dict": model.state_dict()}, path + ".pt")
