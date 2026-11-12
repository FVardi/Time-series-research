# FordA (UCR archive)

**Source:** UCR archive, timeseriesclassification.com, `FordA_TRAIN.txt` / `FordA_TEST.txt`.
Engine noise, 500 samples per series, univariate, two classes (−1, 1). Status `final`.
**Config part:** `configs/datasets/forda.yaml`. **Role:** sanity check against published numbers;
the first slice of the harness (D26, D70).

## Current rules
- Used as distributed: official train/test split (3,601 / 1,320), the archive's per-series
  z-normalisation kept, nothing added (D27, D30). No windows section; FordA is the exception to
  the global standardisation (D255).
- Probes: `svm_ts2vec` (reproduction where a published number exists) and `logreg`; FordA keeps its
  probes without `floor` (user, 2026-11-10); label budgets 1/10/100 % (D26, D29, D40, D133).
- Published references: TS2Vec 0.936, T-Loss 0.928, TS-TCC 0.930, MiniRocket 0.950 (own ridge
  probe), TS-JEPA 0.915 (protocol differs) (D26, D32, D65, D81, D108).
- Comparison list: `configs/compare/forda.yaml` (D203).

## Findings
- Does not separate TS2Vec and T-Loss; appears close to solved by spectral information (D35, D88).
  No training settings are tuned on FordA (D88).

History: D26, D27, D29, D30, D40, D70, D88, D133, D241, D242.
