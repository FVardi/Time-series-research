# TS-TCC

**Source:** Eldele et al., IJCAI 2021; official code github.com/emadeldeen24/TS-TCC, submodule
`external/tstcc` at 15bdfd6. Wrapped (D10 mode A): adapter `harness/methods/tstcc.py` writes out
main.py's self-supervised steps and calls the official `Load_Dataset`, `base_Model`, `TC`,
`model_train` (D104). One epoch of the adapter gives exactly the official weights (FordA, CPU).
**Reported as:** TS-TCC (permutation fixed). **Settings:** `configs/methods/tstcc.yaml`
(official HAR config, checked against the file by the adapter; a difference stops the run).

## Current rules
- Setup of the TS2Vec paper's TS-TCC baseline: official HAR config with 320 output channels and
  batch 128 (D105). Hard-coded official values written out: weight decay 3e-4, λ1 = 1, λ2 = 0.7.
- Representation: encoder output max-pooled over time (D106).
- Permutation fixed for multichannel data: the same segment order for every channel (official:
  channel 0 copied into every channel) (D156). Identical on univariate data.
- Per dataset (shape-forced): `input_channels`, `features_len` (encoder output length; checked by
  the adapter) and `num_classes` (sizes an unused layer) (D156, D173, D256).

## Findings (official code kept as is, D107)
- Augmentations drawn once when the dataset is built, so every epoch sees the same two views.
- The weak view is per-sample scaling only (no separate jitter).
- Number of segments drawn from 1…max_seg − 1; one-segment views are not permuted.
- Temporal contrasting draws one time step per batch.
- Run fix: NumPy ≥ 1.24 refuses the official ragged permutation; the adapter shuffles indices
  (same draws) (D104 note).
- FordA reproduction: 0.931 vs 0.930 published (SVM protocol).

## Dropped
- CA-TCC (D208).

History: D104–D108, D156, D173, D208, D256.
