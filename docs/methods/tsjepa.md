# TS-JEPA

**Source:** Ennadir, Golkar & Sarra, arXiv 2509.25449; official code github.com/Sennadir/TS_JEPA,
submodule `external/tsjepa` at 830ee5c. Wrapped (D10 mode A): the official training loop is a
script, so the adapter `harness/methods/tsjepa.py` writes the loop out and calls the official
Encoder, Predictor, apply_mask, init_weights and loss_pred (tested equal to the script) (D65).
**Reported as:** TS-JEPA (masking fixed); on multichannel data "channels flattened per patch".
**Settings:** `configs/methods/tsjepa.yaml`.

## Current rules
- Masking `per_sample`: the official `apply_mask` gives every series in a batch the first series'
  tokens; fixed, and only the fixed version is compared (D66, D102).
- One encoder layer, 128 dimensions, 2 heads (README; the only source consistent with the paper);
  10 patches, mask ratio 0.7, batch 32, EMA 0.998 (D66).
- Learning rate 6e-5, the official pretraining config's default; no search (D69).
- Epochs: FordA's number of optimiser steps (5,000 epochs × 113 batches = 565,000), capped at
  5,000 epochs: epochs = min(5,000, round(565,000 / batches per epoch)). FordA, CWRU, Herning
  5,000; Paderborn 2,004; HUST 1,290 (D256).
- Representation: mean over the patch vectors (D66).
- Multichannel: each patch holds the same time span of every channel, one after the other; patch
  length per channel rounded down to a multiple of the tokenizer kernel (3), remainder unused
  (D154, D195).
- fp32 (mixed precision measured slower and changed the losses, D256).

## Findings
- FordA reproduction target 0.915 (paper) rests on an unspecified frozen-encoder head; the paper's
  TS2Vec FordA number (86.4) is far below TS2Vec's own (93.6), so the protocols differ (D65).
- FordA standard run `runs/20261016_114616_tsjepa_forda_fixed`: not timed, uncommitted tree;
  not rerun (D102, D210).
- On CWRU the loss drops within 10 epochs, rises to about 0.5 by epoch 500 and drifts to 0.41 (D200).

## Open
- Epoch diagnostic on CWRU (RankMe + validation accuracy every 250 epochs), deferred (D200).

History: D65, D66, D69, D102, D154, D195, D200, D210, D241, D256.
