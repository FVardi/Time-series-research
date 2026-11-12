# HUST bearing data (Hanoi University of Science and Technology)

**Source:** Thuan & Hong (2023, arXiv 2302.12533); Mendeley Data cbv7jyx4p9 v3, CC BY 4.0 (D194).
Converted: the steady-state vibration `data` (51.2 kHz) only; one run per file (99 runs); fault
code, flags, bearing type, load and geometry as capture columns (D223). Status `final`.
**Config part:** `configs/datasets/hust.yaml`.

## Current split and task (D226: the dataset authors' classification experiment)
- Seven classes (N, I, O, B, IO, IB, OB); all 99 recordings (five bearing types, three loads —
  loads inferred).
- 2,048-sample windows with 75 % overlap (hop 512), 2,500 per class drawn at random, stratified
  80/20 split (14,000 / 3,500), `random_state` 0 fixed for all seeds (seed: our choice).
  Overlapping windows of one recording fall on both sides, as in the authors' protocol.
- Pretraining on the training windows; per-channel standardisation; `logreg` (plain stratified
  5-fold), floor; label budgets 1/10/100 % (D226 item 5).

## Not used
- The authors' leave-one-bearing-type-out transfer protocol (D226 alternatives); HUST (Huazhong),
  a different dataset, is on hold (`DATASET_CANDIDATES.md` C21).

History: D189, D190, D194, D223, D226, D243, D256.
