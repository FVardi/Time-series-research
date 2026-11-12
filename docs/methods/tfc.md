# TF-C (time–frequency consistency)

Three things share the name; results are always reported under the specific one (D19).

| name | what | code |
|---|---|---|
| TF-C (official) | 2023 official code, run as distributed | `external/tfc` at 9667582 (github.com/mims-harvard/TFC-pretraining), adapter `harness/methods/tfc_official.py`, `studies/tfc_reproduction/` |
| TF-C (ours) | the paper's method, problems of the official code fixed, paper's loss (`loss.variant: paper`) | `harness/methods/tfc.py` (D41, D42) |
| TF-C (positives) | ours with augmented cross-domain pairs as positives (ablation stage H); the **benchmark version** | `tfc.py` with `loss.variant: positives`, `harness/methods/tfc_frozen.py` for the pipeline (D122, D123) |

**Settings:** `configs/methods/tfc.yaml` (positives; every value with its source).

## Current rules (TF-C positives, benchmark)
- Loss λ(L_T + L_F) + (1 − λ)·(S_TF + S_pair,1 + S_pair,2 + S_pair,3)/4, λ = 0.5, in pretraining
  (and in fine-tuning when fine-tuned) (D122).
- Encoder: paper's CNN for both branches (3 blocks, kernel 8, strides 8/1/1, 32/64/128 channels,
  projector 256 → 128); details the paper lacks from the 2022 code (D41, D124).
- Augmentations redrawn every batch: time — one of jitter (σ 0.1), scaling N(1, 0.1) (our choice),
  permutation (≤ 5 segments); frequency — remove one and add one component (α 0.5),
  symmetric so the spectrum stays that of a real signal (D41, D42).
- Adam lr 3e-4, betas (0.9, 0.99), weight decay 5e-4, batch 64, 40 epochs (D41, D124, D145).
- Frozen-probe pipeline like every method; representation z = [z_T; z_F] (D123).
- Sized from the data on every dataset; FFT per channel (D158, D173).
- Every benchmark dataset gets a TF-C (positives) config (D123).

## Variants under comparison (D202, D217, D227, D243)
`tfc_<dataset>_<variant>.yaml`: `paper` (pairs as negatives, δ = 1; stage E), `left_out`
(C = S_TF; stage D), `left_out_x3` (C = 3·S_TF; stage F), `positives` (standard config), and
λ = 0 arms `left_out_lambda0`, `positives_lambda0` (consistency term only). Run on FordA, CWRU,
HUST, Herning, Paderborn; XJTU-SY to redo.
- **Open:** which variant the benchmark uses — keep positives (D122), switch, or more evidence
  first. So far (seed 0): paper worst and collapsed (RankMe 4–8); λ = 0 as good or best on FordA,
  CWRU, HUST; Paderborn ranks `left_out` first; positives vs left-out not separated.
- Open (D217): in arm `left_out` λ = 0, BatchNorm statistics still see the augmented batches.

## Official code (record of the reproduction, D35–D44, D96)
- Run as distributed except the debugging subset switched off; official protocol (best test
  accuracy) and clean protocol (best validation macro-F1) reported (D37, D43).
- Findings (not fixed, D96): gradients reset once per epoch; the Transformer treats the windows of
  a batch as one sequence (`batch_first=False`); the pretraining loss computes L_C but does not use
  it. Full data: loss NaN in 4 of 5 seeds (D44); with the subset: predicts one class.
- Our TF-C with the paper's loss on FD-A → FD-B: 0.706 ± 0.112 vs 0.8934 published; FD-B is solved
  by the plain spectrum (0.999), so it cannot separate methods (D45, D46, D70).

## Study
Ablation by building blocks and collapse examination: `studies/tfc_ablation/` (D95–D97, D101).

## Open
- FD-A → FD-B transfer with all four variants: pipeline, probes, seeds (D211).
- `tfc.scores` AUROC fails on binary tasks (blocks fine-tuning on two-class data).
- Collapse examination of the official code; `collapse_fd.yaml` placeholders; more seeds for the
  ablation stages (D96).

History: D35–D38, D41–D46, D95–D97, D101, D122–D124, D142, D145, D202, D211, D217, D227, D243.
