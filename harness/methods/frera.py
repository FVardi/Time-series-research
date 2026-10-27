"""FreRA (Tian, Miao & Qian, KDD 2025), official code, wrapped (D10 mode A, D19).

The official code lives unmodified in external/frera (git submodule, commit 7236fbf).
Its main_FreRA.py is a script (pretraining, then a linear classifier) that loads datasets
from fixed paths, so it is not run. Instead, the steps of main_worker() for the SimCLR
framework (the one in the README commands and the paper's main results) are written out
below, in the same order, and the official code does everything inside them:
    FreRA                the learnable frequency augmentation (autoaug/fourier.py)
    SimCLR               FCN encoder + projector + NT-Xent logits (models/builder.py)
    adjust_learning_rate cosine schedule of the SGD learning rate, per epoch
    train                one epoch: view 1 = FreRA(x), view 2 = x unchanged; InfoNCE +
                         l1_weight / len_sw * |w_crit|_1; one step of both optimisers
Left out: the linear classifier (the harness probes the frozen encoder instead) and
fitlog logging (a stand-in object collects the epoch losses that train() reports to it).

What FreRA does to a batch (official, autoaug/fourier.py):
    s = learnable vector, one value per rFFT component (F = L // 2 + 1)
    w_crit = sigmoid((logit(eps) + s) / f_temperature), eps ~ U(0,1): one draw per
             component, shared by every series and channel in the batch
    w_dist = |s_i| / mean(|s_j|, j in D) for i in D = {s_i <= min(0, mean(s))}, else 0 (no gradient)
    view   = irFFT( rFFT(x) * (w_crit + w_dist) )  along time, same weights for every channel

Fixes needed to run (D19: fixes go in the adapter; the official files stay unedited):
  - FreRA.get_sampling moves its noise to the GPU with .cuda(); the copy below moves it to
    the device of the weights instead (same random draws), so it also runs on the CPU.
  - The FCN encoder hard-codes its output length per dataset name (HAR datasets and four UEA
    datasets; no UCR dataset, so FordA does not run). The output length for our series length
    is set as a class attribute FCN.out_len while the model is built; FCN reads it in place of
    the missing per-dataset value. It is checked against the encoder's actual output.
  - main_FreRA.py imports fitlog (a logging package) and data_loaders (dataset files and
    torchvision); neither is used here, so empty stand-ins are put in their place during the import.
Side effect of the official code, kept: importing augmentations.py runs np.random.seed(10);
this happens when the harness imports this module, before scripts/pretrain.py seeds a run.

    pretrain(X, settings, device) -> (model, loss per epoch)
    encode(model, X)              -> one vector per series
    save(model, path)
"""

import sys
import types

import numpy as np
import torch

from harness.methods.external import import_official

NAMES = ("main_FreRA", "models", "autoaug", "augmentations", "data_preprocess")
_stand_ins = {"fitlog": types.ModuleType("fitlog"), "data_loaders": types.ModuleType("data_loaders")}
_saved = {k: sys.modules[k] for k in _stand_ins if k in sys.modules}
sys.modules.update(_stand_ins)
try:
    official_main = import_official("frera", "main_FreRA", local_names=NAMES)  # brings models, autoaug, augmentations
finally:
    for k in _stand_ins:
        del sys.modules[k]
    sys.modules.update(_saved)
FreRA = official_main.FreRA
FCN = official_main.FCN
SimCLR = official_main.models.builder.SimCLR


def get_sampling(self, weight, temperature=0.1, bias=0.0):
    """The official FreRA.get_sampling with one change: `.cuda()` -> `.to(weight.device)`."""
    if self.training:
        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(weight.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(weight.device)  # changed line (official: gate_inputs.cuda())
        gate_inputs = (gate_inputs + weight) / temperature  # todo adaptive temperature
        para = torch.sigmoid(gate_inputs)
        return para
    else:
        return torch.sigmoid(weight)


FreRA.get_sampling = get_sampling


class _LossRecorder:
    """Stands in for fitlog in train(): keeps the epoch's mean loss, ignores everything else."""
    def __init__(self):
        self.losses = []

    def add_loss(self, value, name, step):
        if name == "InfoNCE loss":  # train() reports the mean total loss (InfoNCE + l1 term) under this name
            self.losses.append(float(value))

    def add_metric(self, *args, **kwargs):
        pass


def check_settings(s):
    """Stop if any setting is still an open placeholder (null in the YAML)."""
    for k, v in s.items():
        if v is None:
            raise ValueError(f"frera.{k} is not set (placeholder); see the FreRA decision list")


def fcn_out_len(length):
    """Output length of the FCN encoder: three times conv (kernel 8, padding 4, stride 1),
    which adds one sample, then max-pool (kernel 2, stride 2, padding 1)."""
    for _ in range(3):
        length = (length + 1 + 2 * 1 - 2) // 2 + 1
    return length


def build_model(s, length, n_channels, device):
    """SimCLR as main_worker() builds it (dataset name 'FordA' etc. is unknown to FCN; see fixes above)."""
    FCN.out_len = fcn_out_len(length)
    try:
        model = SimCLR(device, "harness", n_channels, s["batch_size"], "FCN", s["low_dim"], s["temperature"])
    finally:
        del FCN.out_len
    model = model.to(device)
    model.eval()  # check the encoder's actual output length (no dropout, no batch-norm update)
    with torch.no_grad():
        x = torch.zeros(1, n_channels, length, device=device)
        enc = model.encoder_q
        actual = enc.conv_block3(enc.conv_block2(enc.conv_block1(x))).shape[2]
    if actual != fcn_out_len(length):
        raise ValueError(f"FCN output length {actual}, expected {fcn_out_len(length)}")
    return model


def pretrain(X, s, device, train=True):
    """Contrastive pretraining with FreRA as main_worker() does it (framework 'simclr').

    X: array (n_series, n_samples, n_channels), training inputs only. The official UCR loader
    gives (series, time, channels) float32 tensors, shuffled, full batches only.
    s: the 'frera' section of the config; every setting is given explicitly.
    train: False returns the model as built, untrained (random-init baseline, D130).
    Returns {'model', 'aug'} and the mean training loss per epoch.
    """
    check_settings(s)
    if not s["drop_last"]:
        raise ValueError("frera.drop_last must be true: the official NT-Xent mask is built for full batches")
    x = torch.from_numpy(X.astype(np.float32))  # 32-bit (D23)
    dataset = torch.utils.data.TensorDataset(x, torch.zeros(len(X)), torch.zeros(len(X)))  # (sample, label, domain)
    loader = torch.utils.data.DataLoader(dataset, batch_size=s["batch_size"], shuffle=True, drop_last=s["drop_last"])

    # Model, loss and optimisers in the order of main_worker().
    model = build_model(s, X.shape[1], X.shape[2], device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), s["lr"], momentum=s["momentum"], weight_decay=s["weight_decay"])
    aug_f = FreRA(len_sw=X.shape[1], device=device).to(device)
    f = s["f_optimizer"]  # official: AdamW(aug_f.parameters(), lr=f_lr), the rest PyTorch defaults
    f_optimizer = torch.optim.AdamW(aug_f.parameters(), lr=s["f_lr"], betas=tuple(f["betas"]), eps=f["eps"],
                                    weight_decay=f["weight_decay"])
    out = {"model": model, "aug": aug_f, "encode_batch_size": s["encode_batch_size"]}  # trained in place below
    if not train:  # random-init baseline (D130): the models as built above, no training
        return out, []

    # The arguments train() and adjust_learning_rate() read.
    args = types.SimpleNamespace(framework="simclr", f_temperature=s["f_temperature"], l1_weight=s["l1_weight"],
                                 len_sw=X.shape[1], lr=s["lr"], cos=True, epochs=s["epochs"])
    recorder = _LossRecorder()
    for epoch in range(s["epochs"]):
        official_main.adjust_learning_rate(optimizer, epoch, args)  # cosine schedule (args.cos is True in the code)
        official_main.train(epoch, aug_f, f_optimizer, device, loader, model, criterion, optimizer, args, recorder)
    return out, recorder.losses


@torch.no_grad()
def encode(model, X):
    """One vector per series: the flattened encoder output (FCN features before the projector),
    the features the official linear evaluation trains on. Evaluation mode, so batching does not
    change the values."""
    net = model["model"].encoder_q
    net.eval()
    device = next(net.parameters()).device
    x = torch.from_numpy(X.astype(np.float32))
    out = []
    for batch in torch.split(x, model["encode_batch_size"]):
        _, feat = net(batch.to(device), return_feature=True)
        out.append(feat.cpu())
    return torch.cat(out).numpy()


def save(model, path):
    """Save the contrastive model and the learnt frequency weights (main_worker keeps the
    model's state dict; the FreRA weights are logged to fitlog there)."""
    torch.save({"model": model["model"].state_dict(), "aug": model["aug"].state_dict()}, path + ".pt")
