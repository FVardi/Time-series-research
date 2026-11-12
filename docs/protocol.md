# Harness protocol — rules that hold for every dataset and method

Current state, 2026-11-12. Each rule cites the decisions that establish it; superseded
decisions are not listed. Dataset- and method-specific rules and departures are in their cards.

## 1. Data layer

- **P1 Canonical format.** Run → capture → stream → channel; four metadata tables (runs,
  captures, channels, telemetry) as Parquet; one HDF5 file per run, chunks of 16,384 samples,
  gzip 4; signals loaded as float64 (D1–D3, D21, D23). Per-capture facts (labels, conditions,
  official splits) are extra columns on the captures table (D31).
- **P2 As recorded.** No transformation at ingestion; canonical data are immutable (D5).
  Quantized sources are stored as integer codes with scale and offset when every value lies
  exactly on the grid, otherwise as float64, decided per file (D6, D20, D248). Labels,
  resampling, filtering and normalisation happen in the pipeline.
- **P3 Raw data.** Kept unpacked in `data/raw/<dataset>/` in the archive's layout (D213),
  identified by name, size and modification time (D24); never committed (D8). Exception:
  Paderborn RTF keeps no raw vibration files (D235).
- **P4 Conversion.** Follow the dataset's documentation and literature; where they say nothing,
  the harness convention: a run is one test, a capture one uninterrupted acquisition, quirks
  recorded and not fixed (D139, D212, D223, D224, D236, D248, D249, D251).

## 2. Tasks, splits and inputs

- **P5 One split per benchmark dataset**, the first that exists of: official split → the dataset
  authors' protocol → the most widely used protocol with open code (D214, D216). Own data
  (Herning, Jim-Tec) are not bound by this.
- **P6 No test input in pretraining.** Pretraining (and any fitting without labels) uses the
  training windows plus declared pretraining-only data (D164, D225, D231). Checked before
  pretraining at the unit of the split: capture (D33), window (D176) or bearing (D121).
- **P7 Normalisation.** Method and baseline inputs are standardised per channel with one mean and
  SD from the pretraining windows, the same for every window (D137, D253, D255). No per-instance
  normalisation, also where a reference protocol uses it (departures documented: CWRU, XJTU-SY).
  FordA as distributed (D27).
- **P8 Windows.** As the dataset's protocol; otherwise consecutive from the start of each
  recording, the incomplete last window dropped (D137). Set in the dataset config part
  (`windows:`, `split:`; D147).
- **P9 Tasks** follow each dataset's labels: classification, regression (κ, RUL), anomaly
  detection (D13, D192). Unsupervised change detection is not a task (D34).

## 3. Methods

- **P10 Integration.** Official code as pinned git submodules in `external/`, never edited; fixes
  needed to run go in the adapter (`harness/methods/`); a change to the method itself makes a
  separately named method (D10, D19).
- **P11 Settings: the standard, no tuning.** Each method runs with its paper's / official code's
  standard settings; where they disagree, the conflict is decided once per method and recorded in
  its card. No tuning and no search over training length on any dataset (D69, D88, D145, D151).
  Data-dependent official rules are applied to each dataset's data (D151).
- **P12 Only shape-forced changes** between datasets (input channels, lengths and the sizes that
  follow from them), applied by the adapters' formulas (D151, D173, D256). Multichannel
  adaptations are decided once per method and applied everywhere (D154, D155, D156, D159).
- **P13 Every setting in the config** with its source; `null` placeholders stop a run; no silent
  library defaults (D16, CLAUDE.md). Shared settings are in config parts (D260).
- **P14 Failures are recorded, not fixed.** A method that clearly fails to train (loss not falling,
  collapse) is noted case by case (D151); RankMe of the final training representations is logged
  as a diagnostic only (D152). Own methods may use RankMe to choose training length (D146).

## 4. Evaluation

- **P15 Frozen encoders, linear probes.** The standard probe is `logreg`: features standardised
  with training statistics, L2 logistic regression, C by 5-fold CV over 10⁻⁴…10⁴ scored by
  accuracy, `max_iter` 10,000 (D26, D29, D133). Reproduction probes only where a published number
  uses them: `svm_ts2vec` (FordA), `ridge_minirocket` (MiniRocket on FordA) (D26, D81, D133).
- **P16 Regression probe** `ridge`: standardised, α by 5-fold CV over 10⁻⁴…10⁴ scored by R²
  (D58, D59, D170); `ridge_grouped` (XJTU-SY) over 10⁻⁶…10⁸ (D247). Floors: `floor` (most
  frequent class, D83), `floor_mean` (training mean, D170).
- **P17 CV folds.** Grouped by bearing (StratifiedGroupKFold, shuffled per seed) where the split is
  by bearing (D129, D148); plain stratified 5-fold where windows are split at random (D175).
- **P18 Label budgets** 1 %, 10 %, 100 % of the training labels, a new stratified subset per seed
  (regression: unstratified) (D29, D40, D170); a budget is dropped where it leaves fewer labels
  than classes (D178).
- **P19 Test set used once**; every choice (C, α) is made inside the training data (D26).
- **P20 Metrics.** Accuracy and macro-F1; MAE and R² for regression (D26, D59); dataset protocol
  metrics in addition where the protocol has them (e.g. XJTU-SY DSCN score, D231). Anomaly
  detection: 1-NN and Mahalanobis scores on frozen embeddings, AUROC and pAUC (D192; scorers not
  implemented yet).
- **P21 Speed** is measured per stage (fit, encode, probe), on the device the method would use,
  hardware recorded (D79, D81).
- **P22 Baselines** every SSL method must beat: floor; |FFT| (one-sided, linear); squared envelope
  spectrum (bearing data); catch24; MiniRocket; raw samples (SVM probe only, short series);
  domain baselines on Herning (MSSP statistics, pooled band features) (D75, D76, D85, D89, D149,
  D150, D161, D181, D187, D188). Untrained (random-init) encoders as a diagnostic (D134, D135).
  Details: [methods/baselines.md](methods/baselines.md).

## 5. Runs and reporting

- **P23 Standard runs.** One per method and dataset: `configs/<method>_<dataset>.yaml`, the
  dataset's one split, seed 0; only these enter benchmark tables (D239). `--seeds` overrides the
  config's seeds while exploring (D201); more seeds for conclusions later. Baselines use the same
  seeds so differences are paired (D78).
- **P24 Run folders** (`runs/<date_time>_<name>/`, not committed): resolved config and its source
  files, git commit, hardware, window table, representations, results (D17, D33, D260). Runs meant
  for results start from a committed tree; otherwise a warning (D99). Superseded runs move to
  `runs/archive/`; diagnostics keep suffixed names (D239).
- **P25 Comparison tables** from hand-written run lists, `scripts/compare.py`
  `configs/compare/<dataset>.yaml` (D203); statistics across datasets wait for about ten datasets.
- **P26 Studies** live in `studies/<name>/` with a README, import only the harness (D11, D91);
  figures only in notebooks, committed with outputs cleared (D100). Dataset examination in
  `eda/<dataset>/` (D68). The results report is updated only when asked.
- **P27 Environment.** uv, Python 3.14 (D14, D132); plain PyTorch, scikit-learn for probes (D15);
  text files with LF (D238).
- **P28 Simple first; methods alike.** Experiments start in their simplest form and are extended one
  refinement at a time, after the simple version has been looked at and when the user asks;
  proposed refinements are noted as options. Extras beyond the standard runs (longer training,
  checkpoints, seeds, variants) go to every method in a comparison or to none (D263).
