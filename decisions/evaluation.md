# Decisions — evaluation

Evaluation tasks, protocols, probes and metrics. Index: `../DECISIONS.md`.

## 2026-10-06 — D13: Tests are chosen per question and per dataset; full grid within a study; built incrementally
**Decision:** Not every test is run for every method. Each test must answer a stated research question. Tasks (regression, classification, anomaly detection) follow from each dataset's labels; protocols (frozen probe, fine-tuning, transfer to an unseen domain) are used only where they answer a question (e.g. fine-tuning where comparison with published results requires it; transfer only on datasets built for it). Within a study, the test grid is fixed in the study plan before running, and every method fills every cell; a cell a method cannot run is reported as missing with the reason, never dropped. The grid is built incrementally: comparisons are added only when shown to be necessary. The Step 1 grid discussed on 2026-10-06 is preliminary and not decided.
**Alternatives considered:** Running every task and protocol for every method and dataset (full factorial; hundreds of runs, most answering no specific question). Choosing tests per method (invites cherry-picking).
**Rationale:** Keeps comparisons fair and the number of runs proportionate to the questions asked.

## 2026-10-07 — D26: Evaluation protocol for the first slice (TS2Vec on FordA)
**Decision:**
1. *Two probes on the same frozen representations* (one 320-dimensional vector per series, max-pooled over time as in TS2Vec). (a) Reproduction check: TS2Vec's own protocol — RBF-kernel SVM, C chosen by 5-fold cross-validation on the training set over 10⁻⁴…10⁴ and ∞ (exactly as in the official code). (b) Harness probe: logistic regression with L2 regularisation on features standardised with training-set statistics, C chosen by 5-fold cross-validation on the training set over 10⁻⁴…10⁴.
2. *Test set used once.* All choices (probe C) are made by cross-validation inside the training set. No checkpoint selection: TS2Vec trains for a fixed number of iterations.
3. *Metrics:* accuracy and macro-F1.
4. *Seeds:* 3. The seed controls TS2Vec's initialisation and random crops; probes are deterministic.
5. *Label budgets:* 1 %, 10 % and 100 % of the training labels for the probes. Pretraining always uses all training inputs (without labels).
6. *Reproduction criterion:* the published 0.936 accuracy (TS2Vec paper, batch size 8) must lie within our mean ± 2 standard deviations over seeds; if the spread is very small, a difference of up to 1 percentage point is accepted. If it fails, investigate before going further.
**Alternatives considered:** Only the plan's logistic-regression probe (no comparison with the published number). TS2Vec's own logistic regression (no tuning of C). 100 % labels only (label budgets deferred).
**Rationale:** (a) validates our wrapped TS2Vec against the published result (D10); (b) is the harness's standard probe from the research plan. Choosing everything inside the training set keeps the test set clean.

## 2026-10-07 — D29: Label subsets and probe cross-validation details
**Decision:** For label budgets below 100 %, a new stratified random subset of the training labels is drawn for each seed (using that seed), so the spread over seeds includes the effect of which labels were drawn. The logistic-regression probe chooses C by cross-validated accuracy (scikit-learn's default scoring for classifiers), with `max_iter=10000` so the solver converges.
**Alternatives considered:** One fixed subset per label budget shared by all seeds (spread would reflect only pretraining randomness). Macro-F1 as the cross-validation score.
**Rationale:** Reported spread reflects both sources of randomness a user of the method would face; accuracy matches the reproduction metric.

## 2026-10-08 — D34: Change detection is not used as a test
**Decision:** Unsupervised change detection (is the bearing still in its reference state?) is not one of the harness's evaluation tasks. Lubrication condition is evaluated through supervised targets (e.g. κ, discrete lubrication conditions, Stribeck regions, degradation mechanisms), whichever is chosen once the data allow it.
**Alternatives considered:** Change detection as a fourth task type alongside regression, classification and ordinal classification (needs no labels; would fit unlabelled field data).
**Rationale:** User's decision (2026-10-08).

## 2026-10-08 — D40: Label budgets 1 %, 10 % and 100 % throughout Part 1
**Decision:** All Part 1 tests use the three label budgets 1 %, 10 % and 100 % of the training labels for the probes, as in D26 for the first slice; subsets drawn as in D29.
**Alternatives considered:** Fewer budgets outside the FordA slice (fewer runs, but less information on label efficiency, the main argument for SSL).
**Rationale:** User's decision (2026-10-08).

## 2026-10-08 — D45: Frozen-probe diagnostic of our TF-C encoders on FD-B
**Decision:** Diagnostic (`scripts/probe_tfc.py`, `configs/probe_tfc_fd.yaml`): the pretrained encoders of our TF-C run (D41, D42; run `20261008_142751_tfc_fd`, 5 seeds) are frozen; their representation [z_T; z_F] is probed with both harness probes (D26) trained on the 60 FD-B training windows and scored once on the FD-B test set. Label budget: 100 % only, as an exception to D40 for this dataset: 1 % of 60 labels is less than one window, and 10 % (2 per class) is too few for the logistic-regression probe's 5-fold cross-validation. Reference: an untrained encoder of the same architecture, initialised with the same seeds, probed the same way.
**Alternatives considered:** 100 % and 10 % with only the SVM probe at 10 %. A different cross-validation rule for small label sets. No untrained reference (then it cannot be told whether pretraining helped).
**Rationale:** User's decision (2026-10-08). Separates "pretraining learned little" from "fine-tuning too short or selection too noisy" (our TF-C: 0.706 ± 0.112 accuracy vs 0.893 published), without new training.

## 2026-10-08 — D46: Baseline: probes on the plain |FFT| spectra of FD-B
**Decision:** Baseline (`scripts/probe_spectrum.py`, `configs/probe_spectrum_fd.yaml`): both harness probes (D26) trained directly on the full FFT magnitude spectrum of each FD-B training window (the input TF-C's frequency branch receives, D41 item 2), no encoder; 100 % of the 60 training labels (as D45); scored once on the FD-B test set. The probes are deterministic, so there are no seeds.
**Alternatives considered:** No baseline (then it cannot be told whether FD-B separates methods at all).
**Rationale:** User's decision (2026-10-08). The untrained-encoder SVM probe reached 0.973 (D45); if the plain spectra score as high, FD-B cannot separate the methods.

## 2026-10-09 — D49: Feature baseline: the MSSP statistics per octave band (details open) (partly settled by D50)
**Decision:** A hand-crafted feature baseline is added as a method without training (`encode(X)` returns features; the harness probes go on top, logistic regression for classes and a ridge probe for regression, the latter still to be decided). Features: the 14 statistics of the MSSP feature code (`publication-ceramicspeed-dataset/src/ceramicspeed/features.py`: RMS, skewness, kurtosis, crest, shape and margin factor, Hjorth mobility and complexity; dominant frequency, spectral centroid, bandwidth, skewness, kurtosis and flatness), computed per band of an octave filter bank instead of the MSSP pipeline's hand-picked bands. The MSSP pipeline is not reproduced as such (it is unpublished). Open: octave or third-octave, band edges, frequency range, filter implementation, window, and whether a broadband set is kept.
**Alternatives considered:** Octave-band energies only (simpler, but not the features). The MSSP pipeline as it stands (hand-picked bands, rig-specific).
**Rationale:** User's decision (2026-10-09). Same statistics, but a neutral band choice that applies to every dataset.

## 2026-10-09 — D50: Feature baseline details: third-octave bands, Butterworth filters, broadband set kept (band spacing superseded by D53)
**Decision:** For D49: third-octave bands; each band filtered with a zero-phase Butterworth band-pass of order 5 (as `bandpass_filter` in the MSSP code); the 14 statistics are also computed on a broadband version of each signal. Still open: band-edge convention and the frequency range per sensor (the range follows from D52).
**Alternatives considered:** Full octaves (only 3–4 bands in the AE range). FFT-bin summation instead of filters. No broadband set.
**Rationale:** User's decision (2026-10-09).

## 2026-10-09 — D53: Feature baseline uses full-octave bands, base 10, standard edges (bands per sensor superseded by D60)
**Decision:** The filter bank of D49/D50 uses full-octave bands, base 10 (centre frequencies f_m = 1 kHz · 10^(0.3·x), edges f_m · 10^(±0.15)), with the standard edges unchanged (outer bands are not cut at the sensor's range limits). With the current ranges: AE 4 bands (centres 63.1, 126, 251, 501 kHz; 44.7–708 kHz in total); US 8 bands (centres 1.0 … 126 kHz; 0.71–178 kHz in total). Supersedes the third-octave spacing in D50 (third-octave bands give 490 features for AE + US). The frequency range per sensor still follows from D52.
**Alternatives considered:** Third-octave (12 AE / 21 US bands). Half-octave (8 / 15). Cutting the outer bands at the range limits, or dropping bands mostly outside the range.
**Rationale:** User's decision (2026-10-09). Fewer, wider bands: 14 statistics × (bands + broadband) gives 70 AE + 126 US features.

## 2026-10-09 — D54: Feature baseline: four US bands; features over whole sweeps (US bands superseded by D57)
**Decision:** (1) The four lowest US octave bands (centres 1.0–7.94 kHz) are dropped, leaving four US bands (centres 15.8, 31.6, 63.1, 126 kHz; 11.2–178 kHz), the same number as for AE. The US broadband set starts at 0 Hz (upper limit from D52); content below 11.2 kHz is thus only in the broadband statistics. (2) The 14 statistics are computed over whole sweeps (one feature vector per sweep and sensor), as in the MSSP code.
**Alternatives considered:** All eight US bands. Features per window (the unit the SSL methods see).
**Rationale:** User's decision (2026-10-09). The lowest bands are too low; equal band count for both sensors.

## 2026-10-09 — D57: Feature baseline: seven US bands (lowest kept, highest dropped) (bands per sensor superseded by D60)
**Decision:** US octave bands: centres 1.0, 2.0, 3.98, 7.94, 15.8, 31.6, 63.1 kHz (0.71–89.1 kHz); the 126 kHz band is dropped. Supersedes part (1) of D54. AE keeps its four bands (D53).
**Alternatives considered:** Four US bands 15.8–126 kHz (D54): the strongest US content (below 8 kHz, MSSP config) would only enter through the broadband statistics.
**Rationale:** User's decision (2026-10-09).

## 2026-10-09 — D58: Ridge probe for regression, regularisation chosen on a grid
**Decision:** Regression targets (κ) are probed with ridge regression; the regularisation strength is chosen by cross-validation over a fixed grid (deterministic, like the logistic probe, D26), not by Optuna. The label budgets of D40 (1 %, 10 %, 100 %) also apply to regression. Open: grid range, cross-validation grouping on the rig (D56), standardisation and metrics (research plan: MAE and R²).
**Alternatives considered:** Optuna (useful for several hyperparameters, here only one; adds a seed).
**Rationale:** User's decision (2026-10-09).

## 2026-10-09 — D59: Ridge probe details
**Decision:** For D58: features standardised with training-set statistics; regularisation strength chosen by cross-validation over the grid 10⁻⁴ … 10⁴ (powers of ten, as the logistic probe's C grid, D26); on the rig, cross-validation folds grouped by acquisition hold with "twin" holds merged (as the MSSP `grouping.py`); metrics MAE and R² (research plan).
**Alternatives considered:** Ungrouped folds (near-duplicate sweeps of one hold in both fold parts bias the choice of regularisation).
**Rationale:** User's decision (2026-10-09).

## 2026-10-09 — D60: Feature bands per sensor after the band check (D52) (AE bands superseded by D61)
**Decision:** AE: nothing below 50 kHz is used (its origin is unknown; the AE coupler high-passes at 50 kHz), and the top octave band (centre 501 kHz, 355–708 kHz) is excluded as mostly noise. AE bands: centres 63.1, 126, 251 kHz (edges 44.7–89.1, 89.1–178, 178–355 kHz). US: the 126 kHz band (89.1–178 kHz) is included again, because the band check shows rpm-dependent content there. US bands: centres 1.0, 2.0, 3.98, 7.94, 15.8, 31.6, 63.1, 126 kHz (0.71–178 kHz). Supersedes the band lists of D53 (AE) and D57 (US). Open: the lowest AE band's edge at 44.7 kHz lies below 50 kHz; the AE broadband range (50 kHz up to 355 kHz, or up to about 560 kHz where the band check still shows signal).
**Alternatives considered:** AE bands from D53 (including 501 kHz); US bands from D57 (up to 89.1 kHz).
**Rationale:** User's decision (2026-10-09), based on `runs/20261009_085307_band_check_herning`. Caveat: whether US content above about 40 kHz is bearing signal or electrical interference is open (QUESTIONS.md, Q3).

## 2026-10-10 — D61: AE feature bands as octaves from 50 kHz, top band ending at 560 kHz
**Decision:** AE bands start at exactly 50 kHz and double: 50–100, 100–200, 200–400 kHz, plus a shortened top band 400–560 kHz, ending where the band check (D52) shows the AE signal stops. AE broadband set: 50–560 kHz (the range these bands cover). An exception to the base-10 standard edges of D53 for AE; US keeps its base-10 bands (D60). Supersedes the AE bands of D60.
**Alternatives considered:** Three bands 50–400 kHz (drops the weak signal 400–560 kHz). Top band ending at 600 kHz (MSSP config limit). Base-10 bands (lowest band edge at 44.7 kHz, below the 50 kHz limit).
**Rationale:** User's decision (2026-10-10). Nothing below 50 kHz (unknown origin), nothing above about 560 kHz (noise, D52).

## 2026-10-16 — D70: Datasets for the Step 1 comparison; FD-A/FD-B only for the TF-C reproduction
**Decision:** The Step 1 comparison of methods uses FordA (sanity check against published numbers; classification), the raw Paderborn bearing data split by bearing (leakage-safe vibration faults, including a held-out operating condition, research plan 1.4; classification), and the Herning rig data (target data; κ regression). Within the study, every method is run on every one of these datasets (D13); a method that technically cannot run on a dataset is reported as missing with the reason. FD-A/FD-B (TF-C's windows, D39) are used only as the record of the TF-C reproduction (D35–D44), not as a benchmark: FD-B is solved by the plain spectrum (0.999, D46) and its leakage cannot be checked (D39). Open: FordB (cheap second reproduction check; adds little scientifically) and CWRU (comparability with the literature; few recordings per fault and load), see QUESTIONS.md Q4.
**Alternatives considered:** Applying every method to every dataset and protocol, including FD-A/FD-B and their transfer protocol (protocols chosen by the papers, not by the research questions). FD-A/FD-B as plain classification like FordA (trivial, leakage unknown).
**Rationale:** User's decision (2026-10-16). Datasets and tasks follow the research questions; the transfer that matters (an unseen operating condition) is tested cleanly on Paderborn and, as a temperature hold-out, on Herning.

## 2026-10-17 — D73: Collapse diagnostics, logged every epoch (open settings and dead-dimension threshold settled by D74)
**Decision:** During pretraining, after every epoch, representations of a fixed set of windows are computed in evaluation mode and the following are written to `collapse.csv` in the run folder (one row per epoch and representation): effective rank (RankMe: exp of the entropy of the normalised singular values); the full eigenvalue spectrum of the covariance; the mean per-dimension standard deviation and the fraction of dimensions with standard deviation below ε; the mean pairwise cosine similarity; the SIGReg statistic (LeJEPA, arXiv 2511.08544), computed without gradients and not used in the loss. The functions live in `harness/diagnostics.py` (D71). The measures only detect collapse; nothing is enforced. Open: which representations are logged (h_T, h_F, z_T, z_F, [z_T; z_F]; augmented views or not; alignment between a view and its augmented version), the value of ε, the SIGReg settings (number of directions, t-grid, weighting; to be taken from the official LeJEPA code), and the window set (first validation run on FordA, details to be decided).
**Alternatives considered:** Effective rank only (cheaper, but does not show how the spectrum collapses or distinguish complete from dimensional collapse). Regularising against collapse straight away (VICReg or SIGReg in the loss; changes the method before its behaviour is known). Logging less often (may miss early collapse).
**Rationale:** User's decision (2026-10-17). TF-C's encoders sometimes collapse; logging the measures per representation shows where and when collapse starts (encoder vs projector, time vs frequency branch) before deciding whether to counter it.

## 2026-10-17 — D74: Collapse diagnostics settings; relative threshold for dead dimensions; FordA validation run
**Decision:** Settings of the collapse diagnostics (D73), for all collapse runs: all five representations are measured (h_T, h_F, z_T, z_F and [z_T; z_F]); alignment is measured (cosine similarity between each window and its augmented version; one augmented view per window, drawn once with the diagnostics seed); diagnostics seed 0; RankMe epsilon 1e-7 (our choice; the paper gives no value); SIGReg with the official LeJEPA settings t_max 3 and 17 points, 1,024 slices (official README; MINIMAL.md uses 256), and the same directions in every epoch (the official code draws new ones every call). A dimension counts as dead if its standard deviation is below 1 % of the largest standard deviation of that representation (replaces the absolute threshold ε of D73). The eigenvalue spectra are written to `spectra.csv` and the parts of the loss (L_T, L_F, L_C, S_TF) to `pretrain_loss.csv`; epoch 0 (untrained model) is measured. Validation run on FordA: one seed (0), all 3,601 training windows as the diagnostic set, TF-C settings as in `configs/tfc_fd.yaml` (D41, D42). Open for FD-A: seeds and the number of diagnostic windows (h has 10,496 dimensions).
**Alternatives considered:** Absolute threshold (D73's ε; not comparable across representations: in a test run the mean norm of h_F was about 70, that of z_T about 0.6). 256 slices. New SIGReg directions every epoch (adds noise between epochs). Without alignment (cannot tell an encoder that is invariant and spread out from one that is invariant because it is constant).
**Rationale:** User's decision (2026-10-17), following Claude's suggestions.
