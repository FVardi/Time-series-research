# CWRU bearing data

**Source:** Case Western Reserve University Bearing Data Center. All files converted, one run per
`.mat` file, per-capture fault/bearing columns, speed as recorded (table speed marked as estimated
where needed), units `unknown` (literature: g) (D116, D139–D141). Status `final`.
**Config part:** `configs/datasets/cwru.yaml`.

## Current split and task (D218: the one split)
The Zhao et al. (2020) benchmark protocol (open code `CNN_Datasets/R_NA/datasets/CWRU.py`) (D167):
- ten classes: healthy; inner race, ball, outer race @6 at 7, 14, 21 mil; drive end only; 0 hp
  (files 97, 105, 118, 130, 169, 185, 197, 209, 222, 234);
- 1,024-sample windows without overlap from the start of each file;
- stratified random 80/20 split of the windows, `random_state=40`, fixed for every seed (D174).

## Inputs and evaluation
- Healthy file 97 (48 kHz) resampled to 12 kHz (`resample_poly`) — departure from Zhao et al., who
  cut it unresampled (59 instead of 238 healthy windows) (D167).
- Per-channel standardisation with the training windows' statistics — departure from Zhao et al.'s
  per-window normalisation (D177, D253).
- Pretraining on the training windows only (D164); leakage check per window (D176).
- `logreg` (plain stratified 5-fold) and floor; label budgets 10 % and 100 % (1 % has fewer labels
  than classes); accuracy also per class (D168, D178, D179).

## Findings
- A healthy class taken from the 48 kHz normal files is recognisable by its recording (5.7–6 kHz
  band power after resampling) (D168).
- \|FFT\| reaches 1.00 on this protocol (D200 note).
- Earlier splits — by bearing (D128) and random windows over five fault classes (D163) — are no
  longer used; their runs are in `runs/archive/` (D218, D240).

## Open
- Random-init copies (D134) and the TS-JEPA epoch diagnostic (D200) on this protocol.

History: D115–D120, D125, D126, D131, D136–D141, D164, D167, D168, D174, D176–D179, D218, D240, D253.
Superseded: D127, D128, D163, D171, D175, D180, D219.
