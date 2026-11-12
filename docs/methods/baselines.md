# Baselines

Methods without training (D77): same interface as the SSL methods, run through
`scripts/pretrain.py` and `scripts/evaluate.py`, probed with the same probes, label subsets and
seeds (paired, D78). Inputs standardised per channel like every method (D255). Code in
`harness/baselines/`; settings in `configs/methods/<name>.yaml`.

| baseline | what | where | decisions |
|---|---|---|---|
| floor | most frequent training class / training mean (a probe, not a method) | every dataset | D83, D170 |
| raw samples | the window as input; SVM probe only (a linear probe on unaligned samples is at chance) | FordA only (short series) | D75, D89, D149 |
| \|FFT\| | one-sided linear magnitude spectrum per channel, channels concatenated | every dataset except Herning | D75, D76 |
| envelope | squared envelope spectrum over the whole band (Hilbert), otherwise as \|FFT\| | bearing datasets | D161 |
| catch24 | catch22 + mean and SD, `pycatch22==1.0.2`, per channel; NaN stops the run | every dataset except Herning | D85, D90, D188 |
| MiniRocket | official code `external/minirocket` at 0b1c245, 10,000 features, 32 dilations; multivariate code on multichannel data; seeded through numba | every dataset | D76, D81, D150 |
| MSSP statistics | the 14 MSSP statistics per octave band and broadband over the whole sweep (MSSP code copied unchanged) | Herning | D49–D61, D82, D181 |
| pooled band features | mean and SD of each per-window band feature over a sweep | Herning | D166, D187 |
| random-init encoders | each SSL method's untrained encoder (same seed as the trained run), as built and with batch-norm statistics from the training data; logreg only | diagnostic: FordA done; CWRU, Paderborn to copy | D134, D135 |

## Reproduction checks
MiniRocket on FordA with its own ridge probe (`ridge_minirocket`, target 0.950) (D81, D133).

## Findings
- FD-B and CWRU (Zhao protocol) are solved by \|FFT\| (0.999, D46; 1.00, D200 note); FordA appears
  close to solved by spectral information (D88).
- On TF-JEPA and SimMTM, the trained FordA encoders were not better than their untrained ones (D142).

## Dropped
LightGBM (D233). Supervised-from-scratch models and non-linear probes are deferred (D75).

History: D46, D75–D90, D134, D135, D149, D150, D161, D166, D181, D187, D188, D233, D255.
