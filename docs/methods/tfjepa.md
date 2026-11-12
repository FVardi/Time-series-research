# TF-JEPA

**Source:** Chaykowsky, ICLR 2026 TSALM workshop; official code github.com/jchaykow/tf-jepa,
submodule `external/tfjepa` at fda93c1. Wrapped (D10 mode A): adapter `harness/methods/tfjepa.py`
runs the loop of `src/train.py:pretrain_tf_jepa` with the official `Load_Dataset`, `TF_JEPA`,
`cosine_byol` (D109).
**Reported as:** TF-JEPA (single pass); on multichannel data "channel-independent".
**Settings:** `configs/methods/tfjepa.yaml`.

## Current rules
- One forward pass per step on the clean series and its |FFT|, with the paper's loss (the official
  second pass on `x.flip(1)` repeats the same input for one channel) (D113).
- drop_last true (four of five official configs) (D113); batch 128, 20 epochs (official
  configs and README's longer schedule; main.py default 5) (D114, D145).
- Representation: `z` = [z_time; z_freq], the online projector outputs (D113).
- Multichannel: each channel through the same model, representations concatenated; model width =
  window length (one token per series) (D155, D173).
- Saves only the weights `encode()` uses (D237).

## Findings
- The paper's jitter and frequency augmentations are not used by the official pretraining (D110).
- 40 epochs gave a modest gain on FordA; not adopted (D142, D145). logreg, mean of 3 seeds at 1 / 10 / 100 %:
  0.815 / 0.868 / 0.911 (20 epochs) → 0.832 / 0.882 / 0.919 (40 epochs). Not used as extra points in comparisons (D263).
- Not configured on Paderborn, HUST and XJTU-SY: width = window length gives about 1.7 B
  parameters with the EMA targets on Paderborn, about 34 B on XJTU-SY (D256). A scalable
  re-implementation of its idea is postponed.

History: D109, D110, D113, D114, D142, D145, D155, D173, D237, D256.
