"""Random-init baseline (D134): untrained encoders, frozen and probed like the trained ones.

The encoder is the one the method's pretrain() builds, returned before any training
(pretrain(..., train=False)). With the same seed it has the initial weights of the trained run.
Two variants (config key random_init):
  as_initialised : (a) the encoder exactly as built.
  bn_from_train  : (b) as (a), but each batch-norm layer's stored mean and variance are set to those
                   of its input over the training series. No weight is changed, no gradient is used.

Why (b): in evaluation mode a batch-norm layer normalises with its stored statistics. A new layer
stores mean 0 and variance 1 whatever the data, so activations can grow or shrink from layer to
layer. (b) separates "untrained weights" from "normalisation not adapted to the data".
"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm

VARIANTS = ("as_initialised", "bn_from_train")


def batchnorm_layers(model):
    """The batch-norm layers of a model as the adapters return it: a torch module, or a dict
    holding torch modules (TS-TCC, FreRA, TF-C). Each layer once."""
    if isinstance(model, torch.nn.Module):
        return [m for m in model.modules() if isinstance(m, _BatchNorm)]
    found = {}
    if isinstance(model, dict):
        for part in model.values():
            for layer in batchnorm_layers(part):
                found[id(layer)] = layer
    return list(found.values())


@torch.no_grad()
def recompute_batchnorm(model, encode, X):
    """Variant (b): set each batch-norm layer's stored mean and variance to the mean and variance
    (population, i.e. divided by n) per channel of its input over all training series X, as the
    method's encode(model, X) computes that input.

    encode() runs the model in evaluation mode, so every layer normalises with its stored
    statistics, and changing one layer's statistics changes the input of the layers after it.
    The pass is therefore repeated once per layer: after pass p, every layer with fewer than p
    batch-norm layers before it has its final statistics, so after len(layers) passes all have.
    Layers that encode() never passes through (e.g. an EMA target branch) keep theirs.
    Returns the number of layers that were set.
    """
    layers = batchnorm_layers(model)
    if not layers:
        raise ValueError("random_init: bn_from_train needs batch-norm layers; this model has none, "
                         "so it would give the same encoder as as_initialised")
    if not all(layer.track_running_stats for layer in layers):
        raise ValueError("a batch-norm layer does not store statistics (track_running_stats=False)")

    for _ in range(len(layers)):
        sums = {}  # layer -> [count, sum, sum of squares], per channel, in 64-bit

        def collect(layer, inputs):
            """Called with each layer's input just before the layer runs (forward pre-hook)."""
            x = inputs[0].double()
            dims = [0] + list(range(2, x.dim()))  # every dimension except the channels (dimension 1)
            s = sums.setdefault(layer, [0, 0.0, 0.0])
            s[0] += x.numel() // x.shape[1]
            s[1] = s[1] + x.sum(dim=dims)
            s[2] = s[2] + (x * x).sum(dim=dims)

        hooks = [layer.register_forward_pre_hook(collect) for layer in layers]
        try:
            encode(model, X)  # representations are discarded; only the layer inputs are wanted
        finally:
            for h in hooks:
                h.remove()
        for layer, (n, total, total_sq) in sums.items():
            mean = total / n
            layer.running_mean.copy_(mean)
            layer.running_var.copy_(total_sq / n - mean ** 2)
    return len(sums)
