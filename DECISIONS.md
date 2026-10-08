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

<!--
Entry template:

## YYYY-MM-DD — Dn: Short title
**Decision:** What was decided.
**Alternatives considered:** Other options and why they were not chosen.
**Rationale:** Why this option.
-->
