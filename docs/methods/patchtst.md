# PatchTST (self-supervised)

**Source:** Nie et al., ICLR 2023 (arXiv 2211.14730); official code github.com/yuqinie98/PatchTST
(`PatchTST_self_supervised`), submodule `external/patchtst` at 204c21e. Wrapped (D10 mode A):
adapter `harness/methods/patchtst.py` repeats `patchtst_pretrain.py`'s steps with the official
model, Learner, RevInCB, PatchMaskCB and SaveModelCB (D73).
**Reported as:** PatchTST (mean pooling). **Settings:** `configs/methods/patchtst.yaml`.

## Current rules
- Patch length and stride 12 (non-overlapping), 100 epochs (paper), learning rate from the official
  learning-rate finder per seed, 12.5 % of the training series held out to keep the epoch with the
  lowest validation loss, d_ff 256 (paper), default size (3 layers, 16 heads, d_model 128),
  mask ratio 0.4, RevIN on (D74, D144).
- Representation: mean over patch vectors per channel, channels concatenated (D87; chosen after
  FordA test results, which can only make the baseline stronger).
- Channel-independent by design; per dataset only the patch count follows from the length, the
  first 8 samples unused where the length is not a multiple of 12 (D158, D173, D256).
- Run fix: newer PyTorch rejects the `verbose` argument of the one-cycle callback (D73).

## Findings
- On FordA the per-seed results follow the learning rate the finder suggests (D144).
- RevIN normalises each window by its own mean and SD, removing window amplitude (D158, D173).
- Collapses on CWRU, Herning and HUST: masked-patch loss stays near 1.0; HUST RankMe 3.7 of 128,
  logreg 0.46 at 100 %. Not investigated, by the user's decision (2026-11-10); runs kept as the
  standard runs.
- Pooling diagnostic on FordA (mean, flatten, last): D86.

History: D73, D74, D86, D87, D88, D144, D158, D173, D256.
