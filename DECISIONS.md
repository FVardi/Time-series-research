# Decision log

Index of approved design decisions. Full entries live in `decisions/`, one file per topic;
dataset-specific decisions go in `decisions/datasets/`. Numbers are unique across files.
Entries are never deleted; a changed decision is marked "Superseded by Dn".

| ID | Date | Decision | File |
|---|---|---|---|
| D1 | 2026-09-30 | Data hierarchy run → capture → stream → channel; windowing downstream | [decisions/data.md](decisions/data.md) |
| D2 | 2026-09-30 | Metadata as four tables | [decisions/data.md](decisions/data.md) |
| D3 | 2026-09-30 | Canonical format — Parquet metadata, HDF5 signals, one file per run | [decisions/data.md](decisions/data.md) |
| D4 | 2026-09-30 | Minimal adapter interface | [decisions/data.md](decisions/data.md) |
| D5 | 2026-09-30 | No transformation at ingestion; immutable canonical data | [decisions/data.md](decisions/data.md) |
| D6 | 2026-09-30 | Store raw integer codes for quantized sources | [decisions/data.md](decisions/data.md) |
| D7 | 2026-09-30 | Dataset status field | [decisions/data.md](decisions/data.md) |
| D8 | 2026-09-30 | Data is never committed | [decisions/repo.md](decisions/repo.md) |
| D9 | 2026-10-02 | Decision log split by topic | [decisions/repo.md](decisions/repo.md) |
| D10 | 2026-10-03 | Two ways to integrate a method, one shared encoder boundary | [decisions/training.md](decisions/training.md) |
| D11 | 2026-10-03 | One repository; studies in `studies/`; one-way imports | [decisions/repo.md](decisions/repo.md) |
| D12 | 2026-10-03 | κ computed as in the MSSP paper | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D13 | 2026-10-06 | Tests chosen per question and dataset; full grid within a study; built incrementally | [decisions/evaluation.md](decisions/evaluation.md) |
| D14 | 2026-10-06 | Python environments with uv; Python 3.12 (Python version superseded by D132) | [decisions/repo.md](decisions/repo.md) |
| D15 | 2026-10-06 | Plain PyTorch; scikit-learn for probes and simple baselines | [decisions/training.md](decisions/training.md) |
| D16 | 2026-10-06 | One plain YAML config file per experiment | [decisions/repo.md](decisions/repo.md) |
| D17 | 2026-10-06 | Run results as plain files | [decisions/repo.md](decisions/repo.md) |
| D18 | 2026-10-06 | Package layout | [decisions/repo.md](decisions/repo.md) |
| D19 | 2026-10-06 | Official method code as pinned git submodules; modified versions as separate, named methods | [decisions/training.md](decisions/training.md) |
| D20 | 2026-10-06 | Herning scope data stored as 16-bit integer codes | [decisions/data.md](decisions/data.md) |
| D21 | 2026-10-06 | HDF5 chunk size of 16,384 samples | [decisions/data.md](decisions/data.md) |
| D22 | 2026-10-07 | Herning conversion details | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D23 | 2026-10-07 | Compression and loaded data type | [decisions/data.md](decisions/data.md) |
| D24 | 2026-10-07 | Source files identified by name, size and modification time (supersedes checksum part of D5) | [decisions/data.md](decisions/data.md) |
| D25 | 2026-10-07 | Only AE and ultrasound converted for now (supersedes part of D22) | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D26 | 2026-10-07 | Evaluation protocol for the first slice (TS2Vec on FordA) | [decisions/evaluation.md](decisions/evaluation.md) |
| D27 | 2026-10-07 | FordA used as distributed | [decisions/datasets/forda.md](decisions/datasets/forda.md) |
| D28 | 2026-10-07 | TS2Vec training settings for the first slice | [decisions/training.md](decisions/training.md) |
| D29 | 2026-10-07 | Label subsets and probe cross-validation details | [decisions/evaluation.md](decisions/evaluation.md) |
| D30 | 2026-10-07 | FordA metadata | [decisions/datasets/forda.md](decisions/datasets/forda.md) |
| D31 | 2026-10-07 | Datasets may add per-capture columns to the captures table | [decisions/data.md](decisions/data.md) |
| D32 | 2026-10-08 | T-Loss on FordA | [decisions/training.md](decisions/training.md) |
| D33 | 2026-10-08 | Representations saved with a window table; overlap check before pretraining | [decisions/repo.md](decisions/repo.md) |
| D34 | 2026-10-08 | Change detection is not used as a test | [decisions/evaluation.md](decisions/evaluation.md) |
| D35 | 2026-10-08 | TF-C study sequence: reproduce, full Paderborn, own corrected version, possibly FordA | [decisions/training.md](decisions/training.md) |
| D36 | 2026-10-08 | Official TF-C reference = 2023 code (commit 9667582); our own corrected TF-C alongside | [decisions/training.md](decisions/training.md) |
| D37 | 2026-10-08 | Official TF-C run without debugging subset; official and clean protocol reported | [decisions/training.md](decisions/training.md) |
| D38 | 2026-10-08 | TF-C reproduction target (FD-A → FD-B, 0.8934) and 5 seeds | [decisions/training.md](decisions/training.md) |
| D39 | 2026-10-08 | FD-A and FD-B used as distributed by the TF-C authors | [decisions/datasets/fd.md](decisions/datasets/fd.md) |
| D40 | 2026-10-08 | Label budgets 1 %, 10 % and 100 % throughout Part 1 | [decisions/evaluation.md](decisions/evaluation.md) |
| D41 | 2026-10-08 | Design of our own TF-C (paper CNN and loss, corrected symmetric frequency augmentation, selection by validation F1); time-augmentation parameters open | [decisions/training.md](decisions/training.md) |
| D42 | 2026-10-08 | Our TF-C time augmentations: jitter, scaling, permutation (no time shift) | [decisions/training.md](decisions/training.md) |
| D43 | 2026-10-08 | Official TF-C clean protocol selects by validation macro-F1 (supersedes part of D37) | [decisions/training.md](decisions/training.md) |
| D44 | 2026-10-08 | Official TF-C divergence recorded per seed; extra run with the debugging subset | [decisions/training.md](decisions/training.md) |
| D45 | 2026-10-08 | Frozen-probe diagnostic of our TF-C encoders on FD-B (100 % labels, untrained reference) | [decisions/evaluation.md](decisions/evaluation.md) |
| D46 | 2026-10-08 | Baseline: harness probes on the plain FFT magnitude spectra of FD-B | [decisions/evaluation.md](decisions/evaluation.md) |
| D47 | 2026-10-08 | Download all 32 Paderborn bearing archives into data/raw/paderborn | [decisions/datasets/paderborn.md](decisions/datasets/paderborn.md) |
| D48 | 2026-10-08 | Paderborn: convert vibration and both motor currents (64 kHz) | [decisions/datasets/paderborn.md](decisions/datasets/paderborn.md) |
| D49 | 2026-10-09 | Feature baseline: MSSP statistics per octave band (details open) | [decisions/evaluation.md](decisions/evaluation.md) |
| D50 | 2026-10-09 | Feature baseline: third-octave bands, order-5 zero-phase Butterworth, broadband set kept | [decisions/evaluation.md](decisions/evaluation.md) |
| D51 | 2026-10-09 | Rig input: band-limit and downsample per sensor; cut-offs after an information check (superseded by D165) | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D52 | 2026-10-09 | Herning: per-band check of signal vs noise and dependence on operating point, before downsampling | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D53 | 2026-10-09 | Feature baseline: full-octave bands, base 10, standard edges (supersedes D50 spacing) | [decisions/evaluation.md](decisions/evaluation.md) |
| D54 | 2026-10-09 | Feature baseline: four US bands (lowest four dropped); features over whole sweeps | [decisions/evaluation.md](decisions/evaluation.md) |
| D55 | 2026-10-09 | Herning rpm, temperature, exclusions and κ as in the MSSP code; target κ; no classification yet | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D56 | 2026-10-09 | Herning split: temperature hold-out (details open) | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D57 | 2026-10-09 | Feature baseline: seven US bands 1.0–63.1 kHz (supersedes D54 US bands) | [decisions/evaluation.md](decisions/evaluation.md) |
| D58 | 2026-10-09 | Ridge probe for regression, regularisation on a grid; label budgets apply | [decisions/evaluation.md](decisions/evaluation.md) |
| D59 | 2026-10-09 | Ridge probe: standardised, grid 1e-4..1e4, folds grouped by hold, MAE and R² | [decisions/evaluation.md](decisions/evaluation.md) |
| D60 | 2026-10-09 | Feature bands after band check: AE 63.1/126/251 kHz (≥50 kHz, top band out); US 1.0–126 kHz | [decisions/evaluation.md](decisions/evaluation.md) |
| D61 | 2026-10-10 | AE feature bands: 50–100, 100–200, 200–400, 400–560 kHz; AE broadband 50–560 kHz | [decisions/evaluation.md](decisions/evaluation.md) |
| D62 | 2026-10-10 | Downsampling: AE ×2 (1.25 MHz), US ×5 (500 kHz); filter and placement open | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D63 | 2026-10-12 | Results document `report/results.tex`; numbers generated by `scripts/make_report.py`; entries R1, R2, … with status (superseded by D98) | [decisions/repo.md](decisions/repo.md) |
| D64 | 2026-10-12 | Downsampling on the fly; AE and US both to 1.25 MHz (supersedes D62 US factor) (not used for the SSL input since D165) | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D65 | 2026-10-12 | TS-JEPA wrapped from official code (830ee5c); masking, epochs, layers, lr, pooling open | [decisions/training.md](decisions/training.md) |
| D66 | 2026-10-13 | TS-JEPA: both masking variants, 5,000 epochs, 1 encoder layer, mean pooling; lr open (item 1 partly superseded by D102) | [decisions/training.md](decisions/training.md) |
| D67 | 2026-10-14 | TS-JEPA learning rate: search 1e-3..1e-6, fixed variant, 80/20 split of FordA train, logistic-regression probe, seed 0 | [decisions/training.md](decisions/training.md) |
| D68 | 2026-10-14 | Dataset examination code (EDA notebooks, one-off checks) in `eda/<dataset>/`; Herning checks moved there; uv group `eda`; notebooks committed without outputs | [decisions/repo.md](decisions/repo.md) |
| D69 | 2026-10-16 | TS-JEPA learning rate 6e-5 (official default), no search (supersedes D67) | [decisions/training.md](decisions/training.md) |
| D70 | 2026-10-16 | Step 1 datasets: FordA, Paderborn (split by bearing), Herning; FD-A/FD-B only for the TF-C reproduction; FordB, CWRU open (CWRU settled by D115) | [decisions/evaluation.md](decisions/evaluation.md) |
| D71 | 2026-10-17 | TF-C study as an ablation of the time–frequency consistency idea; separate from the baseline list; standalone runs for now (superseded by D95) | [decisions/training.md](decisions/training.md) |
| D72 | 2026-10-17 | TF-C ablation first on simple datasets: FordA, in-domain; Herning later, unfiltered first (superseded by D95) | [decisions/training.md](decisions/training.md) |
| D73 | 2026-10-19 | PatchTST (self-supervised) wrapped from the official code (204c21e); official Learner and callbacks; any number of channels | [decisions/training.md](decisions/training.md) |
| D74 | 2026-10-19 | PatchTST settings: patch 12/stride 12, 100 epochs, official lr finder, 12.5 % validation hold-out, d_ff 256, default size, last-patch representation (item 7 superseded by D87) | [decisions/training.md](decisions/training.md) |
| D75 | 2026-10-16 | Baselines: floor, raw samples, \|FFT\|, MSSP statistics, MiniRocket; untrained encoders as control; supervised-from-scratch deferred (first recorded as D69, renumbered; item d partly superseded by D85) | [decisions/evaluation.md](decisions/evaluation.md) |
| D76 | 2026-10-16 | Baseline inputs: raw as recorded; one-sided linear \|FFT\|; MSSP statistics (broadband only without physical sampling rate); MiniRocket official code and settings (item 3 partly superseded by D85) | [decisions/evaluation.md](decisions/evaluation.md) |
| D77 | 2026-10-16 | Baselines as methods without training, through the same pretrain/evaluate scripts | [decisions/evaluation.md](decisions/evaluation.md) |
| D78 | 2026-10-16 | Baselines use the SSL runs' seeds; differences reported paired per seed | [decisions/evaluation.md](decisions/evaluation.md) |
| D79 | 2026-10-16 | Speed: wall-clock per stage (fit, encode, probe), device as used, hardware recorded; TS2Vec/T-Loss FordA rerun with timing | [decisions/evaluation.md](decisions/evaluation.md) |
| D80 | 2026-10-16 | Reporting: paired difference to the best baseline per budget, label-efficiency figure, time columns, new R entry | [decisions/evaluation.md](decisions/evaluation.md) |
| D81 | 2026-10-19 | MiniRocket: own ridge probe via scikit-learn's normalize replacement; target 0.950 (FordA); numba seeded from NumPy; compile time measured at import | [decisions/evaluation.md](decisions/evaluation.md) |
| D82 | 2026-10-19 | MSSP statistics from the MSSP code copied unchanged (antropy); fs = 1 on FordA; both probes, SVM scale caveat (FordA part superseded by D85) | [decisions/evaluation.md](decisions/evaluation.md) |
| D83 | 2026-10-19 | Floor as a probe (most frequent class), run with the raw-samples baseline; regression floor with the ridge probe | [decisions/evaluation.md](decisions/evaluation.md) |
| D84 | 2026-10-19 | Baselines on FordA first, Herning after Q1/Q2, Paderborn once converted; none on FD-B; untrained-encoder control later | [decisions/evaluation.md](decisions/evaluation.md) |
| D85 | 2026-10-19 | catch24 (pycatch22, per channel) as the generic feature baseline on all datasets; MSSP band statistics kept on Herning only | [decisions/evaluation.md](decisions/evaluation.md) |
| D86 | 2026-10-19 | PatchTST pooling diagnostic: saved FordA models re-encoded with mean and flatten pooling, same probes; reported as diagnostic, not as the PatchTST result | [decisions/evaluation.md](decisions/evaluation.md) |
| D87 | 2026-10-19 | PatchTST representation: mean over patch vectors (supersedes D74 item 7); the strongest version is the baseline; FordA result = run 20261019_154248_patchtst_forda_mean | [decisions/training.md](decisions/training.md) |
| D88 | 2026-10-20 | No training-setting tuning on FordA; tuning reserved for harder datasets, chosen on the unlabelled pretraining validation loss; which methods it applies to is open | [decisions/training.md](decisions/training.md) |
| D89 | 2026-10-19 | Raw-samples baseline: floor and SVM probes only; logistic cell not run (series not phase-aligned); raw run only where series are short enough (first recorded as D87, renumbered) | [decisions/evaluation.md](decisions/evaluation.md) |
| D90 | 2026-10-20 | catch24 with pycatch22==1.0.2 (features match hctsa v3.0.0); mask_nans false, run stops on NaN; cloud catch24 results (0.5.0) rerun | [decisions/evaluation.md](decisions/evaluation.md) |
| D91 | 2026-10-17 | `studies/` holds studies and paper material; method-independent diagnostics in `harness/diagnostics.py`; first study `studies/tfc_ablation/` (first recorded as D71, renumbered) | [decisions/repo.md](decisions/repo.md) |
| D92 | 2026-10-17 | TF-C study `studies/tfc_ablation/`: examination of collapse and ablation by building blocks; own training loop; stages open (first recorded as D72, renumbered; superseded by D95) | [decisions/training.md](decisions/training.md) |
| D93 | 2026-10-17 | Collapse diagnostics (effective rank, spectrum, std, cosine similarity, SIGReg statistic) every epoch to `collapse.csv`; settings and threshold in D94 (first recorded as D73, renumbered) | [decisions/evaluation.md](decisions/evaluation.md) |
| D94 | 2026-10-17 | Collapse diagnostics settings (all representations, alignment, 1,024 fixed SIGReg slices); dead dimension = std < 1 % of the largest; FordA validation run: seed 0, all training windows (first recorded as D74, renumbered) | [decisions/evaluation.md](decisions/evaluation.md) |
| D95 | 2026-10-20 | TF-C study (combines D71, D72, D92): ablation of the time–frequency consistency idea in two parts, collapse and building blocks; standalone in `studies/tfc_ablation/`; ablation on FordA in-domain first; collapse also on FD-A (reproduction record) and FordA; Herning later, unfiltered first; open points partly settled by D96, D97 | [decisions/training.md](decisions/training.md) |
| D96 | 2026-10-20 | TF-C study plan: official TF-C measured as is (flaws documented, not fixed); our TF-C as reference; ablation stages A–E (L_T; L_F; both; + S_TF; + L_C); FordA first, then D70 datasets; h for A–C, h and z for D–E; 1 seed first, then up to 3 | [decisions/training.md](decisions/training.md) |
| D97 | 2026-10-20 | Ablation stage details: D weighted like E; C probed on h_T, h_F, [h_T; h_F]; one branch only in A and B; λ = 0.5 pending check of the paper; implementation in `studies/tfc_ablation/ablation.py` | [decisions/training.md](decisions/training.md) |
| D98 | 2026-10-20 | Results document: current results only, by topic, no entry numbers; generated run index of every folder in `runs/`; superseded results leave the text (supersedes D63) | [decisions/repo.md](decisions/repo.md) |
| D99 | 2026-10-20 | Runs meant for the report start from a committed tree; creating a run folder warns when there are uncommitted changes (run still starts) | [decisions/repo.md](decisions/repo.md) |
| D100 | 2026-10-20 | Figures of a study drawn by a notebook in the study folder (functions in a module, runs in a config); inline only, no figure files; committed with outputs cleared | [decisions/repo.md](decisions/repo.md) |
| D101 | 2026-10-21 | Ablation variants: F (E without the pair terms, S_TF weighted as in E), G (hinged L_C), H (augmented pairs as positives); S_pair terms logged separately; note added to D97 (E weights S_TF three times as much as D) | [decisions/training.md](decisions/training.md) |
| D102 | 2026-10-21 | TS-JEPA compared in its masking-fixed version only; official variant not in the comparison (supersedes part of D66 item 1); FordA runs used without timing, rerun deferred | [decisions/training.md](decisions/training.md) |
| D103 | 2026-10-21 | FordA report: TS-JEPA (masking fixed) and PatchTST added from existing runs; "not timed" rows; TS-JEPA reproduction check on the official-masking run (SVM probe) | [decisions/repo.md](decisions/repo.md) |
| D104 | 2026-10-21 | TS-TCC wrapped from the official code (15bdfd6) as `external/tstcc`; CA-TCC deferred with pseudo-labelling | [decisions/training.md](decisions/training.md) |
| D105 | 2026-10-21 | TS-TCC on FordA: TS2Vec paper's setup (HAR config, 320 output channels, batch 128); official config checked by the adapter; target 0.930 | [decisions/training.md](decisions/training.md) |
| D106 | 2026-10-21 | TS-TCC representation: encoder output max-pooled over time | [decisions/evaluation.md](decisions/evaluation.md) |
| D107 | 2026-10-21 | TS-TCC official code used as is; five departures from the paper recorded (fixed views, weak view, channel-0 permutation, segment count, one t per batch) (item 3 fixed by D156) | [decisions/training.md](decisions/training.md) |
| D108 | 2026-10-21 | TS-TCC runs on FordA: seeds 0–2, D26 probes and budgets, timing, reproduction check vs 0.930 | [decisions/evaluation.md](decisions/evaluation.md) |
| D109 | 2026-10-23 | TF-JEPA, SimMTM and FreRA wrapped from the official code (fda93c1, 169513b, 7236fbf); FordA configs with open placeholders; settings open | [decisions/training.md](decisions/training.md) |
| D110 | 2026-10-23 | TF-JEPA pretraining views as in the official code (clean inputs in both passes; augmented views unused) (superseded by D113) | [decisions/training.md](decisions/training.md) |
| D111 | 2026-10-26 | SimMTM batch size 128 on FordA (paper Table 10; code 32); loss curve to be checked at epoch 10 | [decisions/training.md](decisions/training.md) |
| D112 | 2026-10-26 | FreRA l1_weight 0.1, the script's default (as D69); README 0.003 and paper λ recorded as a finding | [decisions/training.md](decisions/training.md) |
| D113 | 2026-10-26 | TF-JEPA: one pass per step, paper's loss (reported as "TF-JEPA (single pass)"; supersedes D110); drop_last true; probes on [z_time; z_freq] | [decisions/training.md](decisions/training.md) |
| D114 | 2026-10-26 | TF-JEPA on FordA: batch size 128 (official configs, paper profiling), 20 epochs (README example; default 5) | [decisions/training.md](decisions/training.md) |
| D115 | 2026-10-23 | CWRU added to the Step 1 datasets (classification); run-to-failure datasets not added; RUL undecided | [decisions/evaluation.md](decisions/evaluation.md) |
| D116 | 2026-10-27 | CWRU: all files converted; subsets chosen per experiment | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D117 | 2026-10-27 | CWRU: 28 mil faults not used in experiments (NTN bearings, other fault depths) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D118 | 2026-10-27 | CWRU: outer-race positions (@3, @6, @12) kept apart, not merged (storage settled by D120) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D119 | 2026-10-27 | CWRU: 40 mil faults not used in experiments (as D117) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D120 | 2026-10-27 | CWRU: outer-race position stored as the recorded clock hour (3, 6, 12) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D121 | 2026-10-27 | Datasets split by bearing: pretraining never sees recordings of test bearings; D33 check at bearing level | [decisions/evaluation.md](decisions/evaluation.md) |
| D122 | 2026-10-27 | Our TF-C for the benchmarks uses stage H's loss (augmented cross-domain pairs as positives) in pretraining and fine-tuning; required setting `tfc.loss.variant` (`paper` / `positives`); study configs and `tfc_fd.yaml` keep `paper`; new `configs/tfc_positives_fd.yaml` | [decisions/training.md](decisions/training.md) |
| D123 | 2026-10-27 | TF-C (positives) on FordA and every later benchmark dataset, same protocol as the other methods; FordA: pretrain.py + evaluate.py (D26 probes, seeds 0–2, timing) via adapter `tfc_frozen`; probes on z = [z_T; z_F]; FordA settings open in `configs/tfc_forda.yaml` | [decisions/training.md](decisions/training.md) |
| D124 | 2026-10-27 | TF-C (positives) settings on FordA as in the FordA ablation (paper/code values, D42 scaling; 40 epochs, batch 64) | [decisions/training.md](decisions/training.md) |
| D125 | 2026-10-27 | CWRU: no healthy class (fault diagnosis only; detection on Paderborn); test bearings = held-out faulty bearings | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D126 | 2026-10-27 | CWRU: normal files used for pretraining (48 kHz; rate matching decided with the pretraining setup) (rate matching settled by D136) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D127 | 2026-10-27 | CWRU split: both ends pooled, five fault classes; one size per class and end held out; repeated random draws (count open) (superseded by D128) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D128 | 2026-10-27 | CWRU split: train on 7 and 14 mil, test on 21 mil (both ends, five classes; fixed; supersedes D127) (sampling rate clarified by D140) (superseded as the main split by D163) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D129 | 2026-10-27 | Datasets split by bearing: logreg probe's CV folds grouped by bearing (StratifiedGroupKFold, 5), shuffled per seed, redrawn if a training part lacks a class (reproduction-probe point settled by D133) (guard narrowed by D148) | [decisions/evaluation.md](decisions/evaluation.md) |
| D130 | 2026-10-27 | `STATUS.md`: living list of what is done, ready and open (takes over from `HANDOFF.md`) | [decisions/repo.md](decisions/repo.md) |
| D131 | 2026-10-27 | CWRU input: DE and FE as one two-channel series; base plate not used; MiniRocket open | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D132 | 2026-10-27 | Harness on Python 3.14, the version the environment runs (supersedes the Python version in D14) | [decisions/repo.md](decisions/repo.md) |
| D133 | 2026-10-27 | logreg is the standard probe (reported everywhere); svm_ts2vec only for reproduction of published numbers; ridge_minirocket only as MiniRocket's FordA reproduction check | [decisions/evaluation.md](decisions/evaluation.md) |
| D134 | 2026-10-27 | Random-init baseline: each method's untrained encoder (initial weights of the trained run, same seed) probed as the trained one; variants (a) as initialised, (b) batch-norm statistics from the training series; configs `*_random.yaml` / `*_random_bn.yaml`; FordA, CWRU, Paderborn (probes superseded by D135) | [decisions/evaluation.md](decisions/evaluation.md) |
| D135 | 2026-10-27 | Random-init runs probed with logreg only (no svm_ts2vec, no reproduction_probe); supersedes the probe part of D134 | [decisions/evaluation.md](decisions/evaluation.md) |
| D136 | 2026-10-27 | CWRU windows: 2,048 samples, no overlap; normal files resampled 48 → 12 kHz (resample_poly) (normalisation and placement: D137) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D137 | 2026-10-27 | CWRU: windows consecutive from file start, last incomplete dropped; each channel standardised with training mean/std (amplitude kept); baselines raw (D76) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D138 | 2026-10-27 | CWRU label budgets as D29 (random windows, stratified by class, per seed); optimism about label diversity noted | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D139 | 2026-10-27 | CWRU conversion: one run per .mat file; per-capture fault and bearing-ID columns; speed as recorded; units unknown; quirks of 99, 277/278, 28 mil files; status final | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D140 | 2026-10-27 | CWRU: experiments use the 12 kHz fault files (48 kHz converted, not used); quirks of 174, 175, 217 | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D141 | 2026-10-27 | CWRU: missing speed replaced by the table speed where a speed is needed, marked as estimated (pipeline step) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D142 | 2026-10-27 | Longer-training check on FordA: TF-JEPA 40, SimMTM 20, TF-C 80 epochs (twice the configs'); judged by probe accuracy | [decisions/training.md](decisions/training.md) |
| D143 | 2026-10-27 | TS2Vec and T-Loss log the loss per iteration/step (official callback; hook on the loss module); FordA runs repeated | [decisions/training.md](decisions/training.md) |
| D144 | 2026-10-27 | PatchTST: learning-rate finder per seed kept (D74 item 3); finding: per-seed results follow the suggested rate | [decisions/training.md](decisions/training.md) |
| D145 | 2026-10-28 | Training length stays at each method's standard setting, no search; D142 longer runs kept as checks (TF-JEPA 40 ep. modest gain, SimMTM 20 no gain, TF-C 80 worse) | [decisions/training.md](decisions/training.md) |
| D146 | 2026-10-28 | Own methods: RankMe on training windows (no labels) as the indicator for training length / checkpoint; details open until the first own method | [decisions/training.md](decisions/training.md) |
| D147 | 2026-10-28 | Windowed inputs: split, windows and standardisation set in the config (harness/windows.py, harness/labels.py); pretrain.py path for configs with 'split' | [decisions/repo.md](decisions/repo.md) |
| D148 | 2026-10-28 | Grouped folds: a class whose labelled windows all come from one bearing is accepted and recorded (narrows D129's guard) | [decisions/evaluation.md](decisions/evaluation.md) |
| D149 | 2026-10-28 | Raw-samples baseline not run on CWRU (only the floor would remain: D89, D133) | [decisions/evaluation.md](decisions/evaluation.md) |
| D150 | 2026-10-28 | MiniRocket on multichannel data: official multivariate code (same commit, official settings); FordA unchanged | [decisions/evaluation.md](decisions/evaluation.md) |
| D151 | 2026-10-28 | CWRU: no tuning; official settings, changed only where the input shape forces it; data-dependent official rules applied | [decisions/training.md](decisions/training.md) |
| D152 | 2026-10-28 | CWRU: RankMe of the final training representations of every method, diagnostic only (never used to choose settings) | [decisions/evaluation.md](decisions/evaluation.md) |
| D153 | 2026-10-28 | TS2Vec and T-Loss on CWRU: FordA settings unchanged (TS2Vec 600 iterations by its data-size rule) | [decisions/training.md](decisions/training.md) |
| D154 | 2026-10-28 | TS-JEPA on multichannel data: channels flattened within each patch (CWRU: 10 patches of 204 samples per channel, last 8 dropped); safety fix for channel count | [decisions/training.md](decisions/training.md) |
| D155 | 2026-10-28 | TF-JEPA on multichannel data: channel-independent (shared model, vectors concatenated); width = window length (2,048) as designed | [decisions/training.md](decisions/training.md) |
| D156 | 2026-10-28 | TS-TCC on CWRU: input_channels 2, features_len 258, num_classes 5; permutation fixed for multichannel data (FordA unchanged) | [decisions/training.md](decisions/training.md) |
| D157 | 2026-10-28 | LightGBM baseline: `min_data_in_leaf = min(20, max(1, n_train // 10))`; features, role and other settings open | [decisions/evaluation.md](decisions/evaluation.md) |
| D158 | 2026-10-28 | PatchTST, FreRA, TF-C (positives) on CWRU: settings unchanged (PatchTST 170 patches; RevIN removes window amplitude, noted) | [decisions/training.md](decisions/training.md) |
| D159 | 2026-10-28 | SimMTM on CWRU: TSlength_aligned 2048, CNNoutput_channel 87; both channels encoded and reconstructed (decoder 2 x 2048) | [decisions/training.md](decisions/training.md) |
| D160 | 2026-10-28 | Datasets split by bearing: test accuracy also listed per test bearing (diagnostic; per_bearing.csv, logreg, every seed and budget) | [decisions/evaluation.md](decisions/evaluation.md) |
| D161 | 2026-10-28 | Squared envelope spectrum baseline: as \|FFT\| except whole-band squared-envelope demodulation first; CWRU now, Paderborn later, not FordA | [decisions/evaluation.md](decisions/evaluation.md) |
| D162 | 2026-10-29 | Submodules tstcc and frera: `ignore = untracked` in .gitmodules (as tloss, minirocket), so __pycache__ no longer marks runs as uncommitted | [decisions/repo.md](decisions/repo.md) |
| D163 | 2026-10-30 | CWRU main split: random windows, every bearing and load on both sides (supersedes D128 as the main split; proportion, probe folds, literature protocol open; healthy class settled by D168) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D164 | 2026-10-30 | CWRU pretraining: training windows and normal files only, never test windows; one fixed test set, one pretraining per seed | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D165 | 2026-10-30 | Herning SSL input: per-window band features (D60/D61 octaves; log energy, Hjorth mobility/complexity, kurtosis as MSSP code; Butterworth at 2.5 MHz; 1 ms windows; 200 × 48 per sweep; standardised with training stats); supersedes D51 | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D166 | 2026-10-30 | Pooled band-feature baseline: the D165 features averaged over a sweep's windows, probed | [decisions/evaluation.md](decisions/evaluation.md) |
| D167 | 2026-10-30 | CWRU literature protocol: Zhao et al. (2020) benchmark (10 classes, drive end, 0 hp, 1,024 windows, stratified random 80/20), healthy file resampled to 12 kHz; normalisation, split seed, budgets open (healthy class and per-class accuracy: D168) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |
| D168 | 2026-10-30 | CWRU healthy class: left out of the main task (D125 stays); kept in the literature protocol, with accuracy also per class (resampling check: band-edge difference recognisable on unseen bearings) | [decisions/datasets/cwru.md](decisions/datasets/cwru.md) |

<!--
Entry template:

## YYYY-MM-DD — Dn: Short title
**Decision:** What was decided.
**Alternatives considered:** Other options and why they were not chosen.
**Rationale:** Why this option.
-->
