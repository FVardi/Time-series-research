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
| D14 | 2026-10-06 | Python environments with uv; Python 3.12 | [decisions/repo.md](decisions/repo.md) |
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
| D51 | 2026-10-09 | Rig input: band-limit and downsample per sensor; cut-offs after an information check | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
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
| D63 | 2026-10-12 | Results document `report/results.tex`; numbers generated by `scripts/make_report.py`; entries R1, R2, … with status | [decisions/repo.md](decisions/repo.md) |
| D64 | 2026-10-12 | Downsampling on the fly; AE and US both to 1.25 MHz (supersedes D62 US factor) | [decisions/datasets/herning.md](decisions/datasets/herning.md) |
| D65 | 2026-10-12 | TS-JEPA wrapped from official code (830ee5c); masking, epochs, layers, lr, pooling open | [decisions/training.md](decisions/training.md) |
| D66 | 2026-10-13 | TS-JEPA: both masking variants, 5,000 epochs, 1 encoder layer, mean pooling; lr open | [decisions/training.md](decisions/training.md) |
| D67 | 2026-10-14 | TS-JEPA learning rate: search 1e-3..1e-6, fixed variant, 80/20 split of FordA train, logistic-regression probe, seed 0 | [decisions/training.md](decisions/training.md) |
| D68 | 2026-10-14 | Dataset examination code (EDA notebooks, one-off checks) in `eda/<dataset>/`; Herning checks moved there; uv group `eda`; notebooks committed without outputs | [decisions/repo.md](decisions/repo.md) |
| D69 | 2026-10-16 | TS-JEPA learning rate 6e-5 (official default), no search (supersedes D67) | [decisions/training.md](decisions/training.md) |
| D70 | 2026-10-16 | Step 1 datasets: FordA, Paderborn (split by bearing), Herning; FD-A/FD-B only for the TF-C reproduction; FordB, CWRU open | [decisions/evaluation.md](decisions/evaluation.md) |
| D71 | 2026-10-17 | `studies/` holds studies and paper material; method-independent diagnostics in `harness/diagnostics.py`; first study `studies/tfc_ablation/` | [decisions/repo.md](decisions/repo.md) |
| D72 | 2026-10-17 | TF-C study `studies/tfc_ablation/`: examination of collapse and ablation by building blocks; own training loop; stages open | [decisions/training.md](decisions/training.md) |
| D73 | 2026-10-17 | Collapse diagnostics (effective rank, spectrum, std, cosine similarity, SIGReg statistic) every epoch to `collapse.csv`; settings and threshold in D74 | [decisions/evaluation.md](decisions/evaluation.md) |
| D74 | 2026-10-17 | Collapse diagnostics settings (all representations, alignment, 1,024 fixed SIGReg slices); dead dimension = std < 1 % of the largest; FordA validation run: seed 0, all training windows | [decisions/evaluation.md](decisions/evaluation.md) |

<!--
Entry template:

## YYYY-MM-DD — Dn: Short title
**Decision:** What was decided.
**Alternatives considered:** Other options and why they were not chosen.
**Rationale:** Why this option.
-->
