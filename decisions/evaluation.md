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
