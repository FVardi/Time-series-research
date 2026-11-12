# XJTU-SY run-to-failure bearing data

**Source:** Wang, Lei, Li & Li (2020, IEEE Trans. Reliab.); 15 bearings (LDK UER204), 3 conditions,
one 1.28 s sample of both accelerometers (25.6 kHz) per minute until failure; no licence stated.
Converted: one run per bearing, one capture per one-minute file, units g, fault element verbatim,
no derived labels (D193, D224). Status `final`.
**Config parts:** `configs/datasets/xjtu_sy.yaml` (RUL), `configs/datasets/xjtu_sy_stage.yaml`
(health stage).

## Current split (D231: DSCN, the dataset authors' protocol)
Per condition, bearings 1–4 train, bearing 5 tests (1_5, 2_5, 3_5); one probe pooled over the three
conditions (our choice); pretraining on every non-test bearing.

## Tasks
- **RUL (regression):** target (N − n)/N, linear from the start, as DSCN (D228). Probe input = the
  representations of the current and 4 previous minutes (DSCN's S = 5, applied to frozen
  representations; minutes 1–4 left out) (D232). `ridge_grouped` (4 folds grouped by bearing,
  α 10⁻⁶…10⁸) and floor_mean; metrics DSCN score and RMSE in minutes plus MAE and R², from half
  of each test bearing's life (D231, D232, D247).
- **Health stage (classification):** healthy before FPT, degrading after; FPT by Li et al. (2015)
  on horizontal kurtosis with a 20-minute reference (our choice) (D229, D230). Test 1_5, 2_5;
  2_4, 3_2, 3_4 pretraining only; 3_5 left out. `logreg` grouped by bearing, floor (D232).
- **Health indicator:** waits for the anomaly scorers (D192, D232).
- Fault element dropped (the split never tests inner race or cage) (D231).

## Inputs
- 12 non-overlapping windows of 2,560 points per one-minute sample (DSCN's input length; last 2,048
  points unused), both channels; pretraining on all windows of the pretraining bearings (8,711 minutes,
  104,532 windows). Probes see one row per minute: the mean of its 12 window representations
  (`pool_per_capture: mean`, our choice); stored Z stay per window (D266, supersedes D254 item 1).
- Per-channel standardisation with pretraining statistics — departure from DSCN's per-sample
  z-score (Eq. 14), which removed the amplitude (FFT no better than the floor) (D254).

## Open
- Input-length test (2026-11-12, diagnostics, FFT baseline, seed 0): first 2,560 points per minute vs whole
  minute. Stage logreg 1/10/100 %: 0.931/0.806/0.711 vs 0.893/0.818/0.824 (floor 0.450). RUL ridge RMSE (min)
  26.6/40.5/21.5 vs 24.7/21.7/20.0, R² at 100 % −0.76 vs −0.74 (floor RMSE 96–105). Runs
  `runs/20261112_11*_fft_xjtu_sy*_first2560`, whole-minute runs in `runs/archive/`. Led to D266.
- Runs to redo under D266 (`scripts/queue_xjtu.ps1`). Methods: TS2Vec, T-Loss, TS-TCC, TS-JEPA (173
  epochs), PatchTST, SimMTM and the TF-C variants on both tasks; FreRA (D264) and TF-JEPA (D256) not run.
- Option, not built (D263): both tasks pretrain on the same 12 bearings, so one pretrained model per
  method could serve both and halve the GPU time.

History: D189, D191, D193, D224, D228–D232, D243, D247, D254, D256, D264, D266.
