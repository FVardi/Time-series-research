# TS2Vec

**Source:** Yue et al., AAAI 2022 (arXiv 2106.10466); official code github.com/zhihanyue/ts2vec,
submodule `external/ts2vec` at b0088e1. Wrapped (D10 mode A): adapter `harness/methods/ts2vec.py`.
**Reported as:** TS2Vec. **Settings:** `configs/methods/ts2vec.yaml` (each value with its source).

## Current rules
- Official defaults: lr 0.001, batch 8, 320 dimensions, hidden 64, depth 10, `max_train_length`
  3,000; iterations by the official rule (200 if the training data has ≤ 100,000 values, else 600)
  applied to each dataset's data (D28, D151, D153).
- Representation: one vector per series, max-pooled over time, as the official evaluation (D26).
- Loss logged per iteration through the official callback; training unchanged (D143).
- Per dataset: no setting depends on the input shape; any channel count and length (D153, D173,
  D256). On Paderborn the 5,120-sample windows are not split (5,120 // 3,000 = 1 section, D256).

## Findings
- Reproduced the published FordA accuracy with the SVM protocol (earlier runs: 0.936 vs 0.936,
  D26); the standard run was made again on 2026-11-10 (D242). FordA does not separate it from T-Loss.
- Loss still falling at 600 iterations; training length not searched (D145).

## Open
- None specific. Further datasets get a config as their protocol is set (P12).

History: D26, D28, D143, D145, D151, D153, D173, D242, D256.
