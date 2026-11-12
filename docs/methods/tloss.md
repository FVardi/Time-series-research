# T-Loss

**Source:** Franceschi, Dieuleveut & Jaggi, NeurIPS 2019; official code
github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries, submodule
`external/tloss` at 4aff592. Wrapped (D10 mode A): adapter `harness/methods/tloss.py` (encoder
training and encoding only; T-Loss's own classifier is not used).
**Reported as:** T-Loss. **Settings:** `configs/methods/tloss.yaml`.

## Current rules
- Official settings: batch 10, lr 0.001, depth 10, 40 channels, kernel 3, K = 10 negatives,
  full-length comparisons, no early stopping, representation 320 (internal 160; the paper swaps
  them), 2,000 steps (paper for K ≥ 10; the shipped default file says 1,500) (D32).
- Trains in float64, as the code requires (D32).
- Loss logged per step through a forward hook; training unchanged (D143).
- Per dataset: no shape-dependent setting (D153, D173, D256).

## Findings
- Reproduced its published FordA accuracy with the SVM protocol (earlier runs: 0.932 vs 0.928,
  D32); the standard run was made again on 2026-11-10 (D242).
- Loss falls slowly at 2,000 steps; not extended (D145).
- Run fixes in the adapter: the official `losses/` and `networks/` packages are stubbed on
  Python ≥ 3.12 (D19).

History: D32, D143, D145, D153, D173, D242, D256.
