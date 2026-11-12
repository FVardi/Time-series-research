# Paderborn bearing data (KAt DataCenter)

**Source:** Lessmeier et al. (2016), KAt-DataCenter, Paderborn University; 32 bearings (6 healthy,
12 artificial, 14 real damage), CC BY-NC 4.0 (D47). Converted: vibration and both motor currents at
64 kHz; one run per bearing and operating condition (128 runs), 20 measurements as captures; labels
from the damage sheets and Lessmeier et al.'s class (D48, D212). Status `final`.
**Config part:** `configs/datasets/paderborn.yaml`.

## Current split and task (D220: the dataset authors' protocol)
Lessmeier et al. (2016) Section 7.3 (Table 10): three classes, five bearings each — healthy
K001–K005, outer ring KA04/15/16/22/30, inner ring KI04/14/16/18/21. Three bearings per class for
training, two for test, 10 combinations (inferred: the C(5, 3) choices of three rows applied to all
classes at once); the score is the mean accuracy over the combinations.
- Setting 0 only (N15_M07_F10), all 20 measurements, vibration channel only, 5,120-sample windows
  without overlap (D222).
- Pretraining on each combination's training windows only, about 9,000 windows (D225).
- Not run here: TF-JEPA (D256) and FreRA (D264), whose official models do not fit on the GPU at 5,120 samples.
- `logreg` with folds grouped by bearing, floor; label budgets 1/10/100 % (D129, D133).
- Run with `scripts/run_combinations.py` (one run per combination, `<name>_c<k>`, plus a
  `<name>_combinations.csv`).

## Open
- Random-init copies (D134). `compare.py` does not read the per-combination runs yet.

History: D47, D48, D121, D129, D212, D213, D220, D222, D225, D227, D256, D264. Superseded: D221.
