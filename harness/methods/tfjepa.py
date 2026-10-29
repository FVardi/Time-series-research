"""TF-JEPA (Chaykowsky, ICLR 2026 TSALM workshop), official code, wrapped (D10 mode A, D19).

The official code lives unmodified in external/tfjepa (git submodule, commit fda93c1).
Its main.py pretrains and then fine-tunes in one script with fixed dataset paths, so it
is not run. The pretraining function src/train.py:pretrain_tf_jepa is short, and it is
written out below so the loss per epoch can be returned; everything inside it comes
from the official code: Load_Dataset (time series, |FFT| and the augmented views),
TF_JEPA (encoders, projectors, EMA targets, predictors) and cosine_byol (the loss).
tests/test_tfjepa.py checks that the loop below (views: official) gives exactly the
weights of the official pretrain_tf_jepa.

What one training step does (official):
    x_t   = the series (channel 0 only, first TSlength_aligned samples; Load_Dataset)
    x_f   = |FFT(x_t)|, full two-sided spectrum, length TSlength_aligned (Load_Dataset)
    out   = model(x_t, x_f): online time/frequency encoders + projectors -> z_t, z_f;
            EMA target encoders + projectors -> targets; predictors t->f and f->t
    loss  = 0.5 * [cos(p_t->f, target_f) + cos(p_f->t, target_t)] summed over two views,
            cos(p, z) = 2 - 2 * cosine similarity (cosine_byol)
    then: optimiser step; EMA update of the target encoders AND projectors (m = momentum)

Views (see `views` in the config): the official loop's second view is
model(x_t.flip(1), x_f.flip(1)); dim 1 is the single channel, so the flip changes
nothing and both views are the clean inputs (they differ only through dropout).
The augmented views that Load_Dataset draws are returned but not used. The paper says
pretraining uses jitter (time) and removing/adding 10 % of frequency bins (frequency)
but not which encoder sees which view. views: augmented uses them as the official
fine-tuning loss does (finetune_tf_jepa: second pass on the augmented time and
frequency inputs); this is not in the official pretraining code. views: single drops
the second pass (D113): one pass on the clean inputs, with the paper's loss
cos(p_t->f, target_f) + cos(p_f->t, target_t), which is what the official loss gives
when both passes see the same inputs (up to dropout). Reported under its own name,
"TF-JEPA (single pass)" (D19).

Several channels (our adaptation, D155; reported as "TF-JEPA (single pass, channel-independent)"):
each channel is passed through the same model as its own univariate series, in pretraining and
in encoding, and a series' representation is its channels' representations concatenated
(channel_series / channel_vectors below). With one channel nothing changes.

    pretrain(X, settings, device) -> (model, loss per epoch)
    encode(model, X)              -> one vector per series
    save(model, path)
"""

import numpy as np
import torch

from harness.methods.external import import_official

# The official files import each other as 'src.model', 'src.loss', 'src.utils'.
NAMES = ("src",)
official_model = import_official("tfjepa", "src.model", local_names=NAMES)
official_loss = import_official("tfjepa", "src.loss", local_names=NAMES)
official_data = import_official("tfjepa", "src.data", local_names=NAMES)


def check_settings(s):
    """Stop if any setting is still an open placeholder (null in the YAML)."""
    def walk(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{path}{k}.")
            elif v is None:
                raise ValueError(f"tfjepa.{path}{k} is not set (placeholder); see the TF-JEPA decision list")
    walk(s, "")


def model_config(s, length):
    """The dict TF_JEPA and Load_Dataset read (official keys; the model's .get() defaults are given explicitly)."""
    return {
        "TSlength_aligned": length,  # d_model of the Transformer = number of samples per series
        "embed_dim": s["encoder"]["embed_dim"],
        "nhead": s["encoder"]["nhead"],
        "nlayers": s["encoder"]["nlayers"],
        "ff_mult": s["encoder"]["ff_mult"],
        "augmentation": {"jitter_ratio": s["jitter_ratio"]},  # σ of the time-domain jitter (Load_Dataset)
    }


def channel_series(X):
    """(series, time, channels) -> (series * channels, time, 1): every channel as its own series,
    series by series (series 0 channel 0, series 0 channel 1, ...) (D155). Unchanged for one channel."""
    n, length, channels = X.shape
    return np.ascontiguousarray(np.transpose(X, (0, 2, 1)).reshape(n * channels, length, 1))


def channel_vectors(Z, channels):
    """(series * channels, d) -> (series, channels * d): a series' channel vectors side by side (D155)."""
    return Z.reshape(len(Z) // channels, channels * Z.shape[1])


def loader(X, s):
    """The official Load_Dataset ('pre_train' mode) in a DataLoader, as data_generator builds it.

    Load_Dataset keeps channel 0 and the first TSlength_aligned samples without saying so;
    here a series with more channels stops the run instead of being cut silently.
    """
    if X.shape[2] != 1:
        raise ValueError("the official TF-JEPA Load_Dataset keeps channel 0 only; the data have "
                         f"{X.shape[2]} channels")
    cfg = model_config(s, X.shape[1])
    samples = torch.from_numpy(np.transpose(X, (0, 2, 1)).astype(np.float32))  # (n, 1, time), 32-bit (D23)
    data = {"samples": samples, "labels": torch.zeros(len(X))}  # labels are not used in pretraining
    dataset = official_data.Load_Dataset(data, cfg, "pre_train")
    return cfg, torch.utils.data.DataLoader(dataset, batch_size=s["batch_size"], shuffle=True,
                                            drop_last=s["drop_last"], num_workers=0)


def pretrain(X, s, device, train=True):
    """Pretrain TF-JEPA on unlabelled training series X (series, time, channels); several
    channels are trained as separate series of one channel (D155).

    s: the 'tfjepa' section of the config; every setting is given explicitly.
    train: False returns the model as built, untrained (random-init baseline, D134).
    Returns the model (online and target branches) and the mean training loss per epoch.
    """
    check_settings(s)
    if s["views"] not in ("official", "augmented", "single"):
        raise ValueError(f"tfjepa.views must be 'official', 'augmented' or 'single', not {s['views']!r}")
    n_channels = X.shape[2]
    cfg, train_loader = loader(channel_series(X), s)  # one channel per series (D155)
    model = official_model.TF_JEPA(cfg, momentum=s["ema_momentum"]).to(device)
    model.train()

    # Optimiser and schedule as in pretrain_tf_jepa (hard-coded there; explicit here).
    o = s["optimizer"]
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=o["lr"],
                            betas=tuple(o["betas"]), eps=o["eps"], weight_decay=o["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s["epochs"])
    cos = official_loss.cosine_byol

    # Settings encode() needs, kept with the model (the harness calls encode(model, X)).
    model.representation, model.encode_batch_size = s["representation"], s["encode_batch_size"]
    model.n_channels = n_channels  # encode() checks that it gets the same channels
    if not train:  # random-init baseline (D134): the model as built above, no training
        return model, []

    log = []
    for epoch in range(1, s["epochs"] + 1):
        losses = []
        for x_t, _, x_t_aug, x_f, x_f_aug in train_loader:
            x_t, x_f = x_t.float().to(device), x_f.float().to(device)
            opt.zero_grad()
            out1 = model(x_t, x_f)
            if s["views"] == "single":
                # One pass (D113): both directions, as the paper's loss.
                loss = cos(out1["p_time_to_freq"], out1["z_freq_target"]) + cos(out1["p_freq_to_time"], out1["z_time_target"])
            else:
                if s["views"] == "official":
                    out2 = model(x_t.flip(1), x_f.flip(1))  # official: flips the single channel = same inputs
                else:
                    out2 = model(x_t_aug.float().to(device), x_f_aug.float().to(device))
                # Official loss: both directions, both passes, halved (symmetric = True in pretrain_tf_jepa).
                loss = cos(out1["p_time_to_freq"], out1["z_freq_target"]) + cos(out2["p_time_to_freq"], out2["z_freq_target"])
                loss += cos(out1["p_freq_to_time"], out1["z_time_target"]) + cos(out2["p_freq_to_time"], out2["z_time_target"])
                loss *= 0.5
            loss.backward()
            opt.step()
            model.update_targets()  # EMA of encoders and projectors (official)
            losses.append(loss.item())
        log.append(float(np.mean(losses)))
        print(f"  epoch {epoch}: loss {log[-1]:.4f}", flush=True)
        sched.step()
    return model, log


@torch.no_grad()
def encode(model, X):
    """One vector per series from the online branches, in evaluation mode (no dropout;
    batch norm uses its stored statistics, so batching does not change the values).

    representation (paper silent; see the config):
      z     : [z_time; z_freq], the two projector outputs (the official classifier's input,
              without the raw-mean feature it adds)
      h     : [h_time; h_freq], the two encoder outputs mean-pooled over tokens, before the projectors
      z_raw : [z_time; z_freq; mean of x_t], exactly the official classifier's input
    """
    model.eval()
    device = next(model.parameters()).device
    if X.shape[2] != model.n_channels:
        raise ValueError(f"the model was trained on {model.n_channels} channel(s), the data have {X.shape[2]}")
    x_t = torch.from_numpy(np.transpose(channel_series(X), (0, 2, 1)).astype(np.float32))  # (series*channels, 1, time)
    out = []
    for batch in torch.split(x_t, model.encode_batch_size):
        batch = batch.to(device)
        x_f = torch.fft.fft(batch).abs()  # as Load_Dataset: |FFT| along time
        h_t = model.enc_time_online(batch).mean(dim=1)  # official _encode: encoder, mean over tokens
        h_f = model.enc_freq_online(x_f).mean(dim=1)
        if model.representation == "h":
            z = torch.cat([h_t, h_f], dim=1)
        elif model.representation in ("z", "z_raw"):
            z = torch.cat([model.proj_time_online(h_t), model.proj_freq_online(h_f)], dim=1)
            if model.representation == "z_raw":
                z = torch.cat([z, batch.mean(dim=-1).view(len(batch), -1)], dim=1)  # official raw_stat
        else:
            raise ValueError(f"tfjepa.representation must be 'z', 'h' or 'z_raw', not {model.representation!r}")
        out.append(z.cpu())
    return channel_vectors(torch.cat(out).numpy(), model.n_channels)  # channels side by side (D155)


def save(model, path):
    """Save all weights as the official main.py saves the pretrained model (a plain state dict)."""
    torch.save(model.state_dict(), path + ".pt")
