"""TS-TCC (Eldele et al., IJCAI 2021), official code, wrapped (D10 mode A, D19, D104).

The official code lives unmodified in external/tstcc (git submodule, commit 15bdfd6).
Its main.py is a script that reads files from fixed paths, so it is not run. Instead,
the steps main.py takes with --training_mode self_supervised are written out below,
in the same order, and the official code does everything inside them:
    Load_Dataset   builds the dataset and draws the weak and strong augmentations
    base_Model     the encoder (3 convolution blocks)
    TC             temporal contrasting (Transformer, predicts future time steps)
    model_train    one epoch: temporal + contextual contrasting loss, backward pass, optimiser steps
Left out from main.py: logging, copying files, and the validation pass after every epoch,
which computes nothing in this mode. (Starting that pass draws one number from PyTorch's
random-number generator, so a run here is not bit-identical to main.py with the same seed
after the first epoch; it is the same procedure.)
Settings: D105. Representation: D106. Properties of the official code kept as they are: D107.
One fix, needed to run with NumPy >= 1.24 (D19: fixes go here, the official files stay
unedited): see permutation() below.

    pretrain(X, settings, device) -> (model, loss per epoch)
    encode(model, X)              -> one vector per series
    save(model, path)
"""

import numpy as np
import torch

from harness.methods.external import import_official

# Top-level names the official repository uses for its own folders.
NAMES = ("models", "dataloader", "trainer", "config_files", "utils")
official_model = import_official("tstcc", "models.model", local_names=NAMES)
official_tc = import_official("tstcc", "models.TC", local_names=NAMES)
official_trainer = import_official("tstcc", "trainer.trainer", local_names=NAMES)
official_dataloader = import_official("tstcc", "dataloader.dataloader", local_names=NAMES)
HARConfig = import_official("tstcc", "config_files.HAR_Configs", local_names=NAMES).Config


def permutation(x, max_segments=5, seg_mode="random"):
    """The official dataloader/augmentations.py permutation(), with one line changed so it runs.

    The official line `np.random.permutation(splits)` shuffles a list of segments of unequal
    length; NumPy >= 1.24 refuses to turn such a list into an array. Shuffling the segment
    indices instead gives the same order from the same random draws (NumPy shuffles k items
    with the same Fisher-Yates loop either way). Everything else is copied unchanged,
    including the use of channel 0 for all channels (D107 item 3).
    """
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            order = np.random.permutation(len(splits))  # changed line (official: np.random.permutation(splits))
            warp = np.concatenate([splits[k] for k in order]).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


# DataTransform (called by Load_Dataset) looks permutation up in the official augmentations
# module; point that name at the fixed copy above.
official_dataloader.DataTransform.__globals__["permutation"] = permutation


def _attribute(configs, name):
    """'augmentation.max_seg' -> (configs.augmentation, 'max_seg'). Stops if it does not exist."""
    obj = configs
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise ValueError(f"{name}: not in the official HAR config")
    return obj, parts[-1]


def official_config(s):
    """The official HAR config with the four values D105 changes.

    s['official_config']: every HAR value used, as written in the official file;
                          a difference stops the run, so nothing is changed silently.
    s['changed']:         the values D105 changes for FordA (input_channels,
                          final_out_channels, features_len, num_classes).
    """
    configs = HARConfig()
    for name, value in s["official_config"].items():
        obj, attr = _attribute(configs, name)
        if getattr(obj, attr) != value:
            raise ValueError(f"{name}: YAML says {value}, official config says {getattr(obj, attr)}")
    for name, value in s["changed"].items():
        obj, attr = _attribute(configs, name)
        setattr(obj, attr, value)
    return configs


def check_sizes(configs, model, X):
    """Stop early with a clear message if the config does not fit the data.

    features_len must equal the encoder's output length: the official model computes its
    (unused) classifier layer on the flattened encoder output in every forward pass.
    """
    if X.shape[2] != configs.input_channels:
        raise ValueError(f"input_channels is {configs.input_channels}, the data have {X.shape[2]} channels")
    was_training = model.training
    model.eval()  # no dropout, no update of the batch-norm statistics
    with torch.no_grad():
        x = torch.zeros(1, X.shape[2], X.shape[1], device=next(model.parameters()).device)
        length = model.conv_block3(model.conv_block2(model.conv_block1(x))).shape[2]
    model.train(was_training)
    if length != configs.features_len:
        raise ValueError(f"features_len is {configs.features_len}, the encoder output has length {length}")


def pretrain(X, s, device, train=True):
    """Self-supervised training as the official main.py does it (--training_mode self_supervised).

    X: array (n_series, n_samples, n_channels), training inputs only.
    s: the 'tstcc' section of the config (D105); every setting is given explicitly.
    train: False returns the model as built, untrained (random-init baseline, D130).
    The seeds are set before this call (scripts/pretrain.py); main.py's other seeding
    lines are repeated here. Returns the model and the mean training loss per epoch.
    """
    configs = official_config(s)
    torch.backends.cudnn.deterministic = False  # as in main.py
    torch.backends.cudnn.benchmark = False

    # The official data_generator(), training part. The official files hold 'samples' as
    # (n, channels, time); conversion to 32-bit happens here, at training (D23).
    # Labels are not used in this mode; zeros fill the slot Load_Dataset expects.
    samples = torch.from_numpy(np.transpose(X, (0, 2, 1)).astype(np.float32))
    data = {"samples": samples, "labels": torch.zeros(len(X), dtype=torch.long)}
    dataset = official_dataloader.Load_Dataset(data, configs, "self_supervised")  # draws both views (D107 item 1)
    if dataset.x_data.shape != samples.shape:  # Load_Dataset guesses which axis holds the channels
        raise ValueError(f"Load_Dataset rearranged the data to {tuple(dataset.x_data.shape)}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True,
                                         drop_last=configs.drop_last, num_workers=0)

    # Models and optimisers as in main.py (weight decay is hard-coded there).
    model = official_model.base_Model(configs).to(device)
    tc = official_tc.TC(configs, device).to(device)
    check_sizes(configs, model, X)
    adam = dict(lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=s["weight_decay"])
    model_optimizer = torch.optim.Adam(model.parameters(), **adam)
    tc_optimizer = torch.optim.Adam(tc.parameters(), **adam)
    if not train:  # random-init baseline (D130): the models as built above, no training
        return {"encoder": model, "tc": tc, "batch_size": configs.batch_size}, []

    # The training loop of the official Trainer(). In this mode the loss is
    # λ1·(both temporal contrasting losses) + λ2·NT-Xent, with λ1 = 1 and λ2 = 0.7
    # hard-coded in model_train. The cross-entropy criterion is passed as Trainer()
    # passes it; this mode does not use it.
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    for epoch in range(1, configs.num_epoch + 1):
        loss, _ = official_trainer.model_train(model, tc, model_optimizer, tc_optimizer, criterion,
                                               loader, configs, device, "self_supervised")
        losses.append(float(loss))
        print(f"  epoch {epoch}: loss {float(loss):.4f}", flush=True)
    return {"encoder": model, "tc": tc, "batch_size": configs.batch_size}, losses


def encode(model, X):
    """One vector per series (D106): the encoder output, max-pooled over time.

    The encoder runs in evaluation mode (no dropout; batch-norm uses its stored statistics),
    so the batch size only affects speed, not the values.
    """
    encoder = model["encoder"]
    encoder.eval()
    device = next(encoder.parameters()).device
    x = torch.from_numpy(np.transpose(X, (0, 2, 1)).astype(np.float32))
    out = []
    with torch.no_grad():
        for batch in torch.split(x, model["batch_size"]):
            _, z = encoder(batch.to(device))  # z: (batch, channels, time), before any normalisation
            out.append(z.max(dim=2).values.cpu().numpy())
    return np.concatenate(out)


def save(model, path):
    """Save the weights in the official checkpoint format (as ckp_last.pt in the official code)."""
    torch.save({"model_state_dict": model["encoder"].state_dict(),
                "temporal_contr_model_state_dict": model["tc"].state_dict()}, path + ".pt")
