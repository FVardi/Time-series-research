# Decisions — repository

Repository structure and project conventions. Index: `../DECISIONS.md`.

## 2026-09-30 — D8: Data is never committed
**Decision:** The `/data/` folder is excluded from git via `.gitignore`.
**Alternatives considered:** Ignoring by file extension (`*.h5`, `*.hdf5`); requires updating for every new format.
**Rationale:** Data files are large and not version-controlled; ignoring the folder covers every format.

## 2026-10-02 — D9: Decision log split by topic
**Decision:** Decisions are recorded in a `decisions/` folder with one file per topic, following the pipeline layers rather than the code layout: `data.md`, `labels.md`, `splits.md`, `training.md`, `evaluation.md`, `repo.md`, and `datasets/<dataset>.md` for dataset-specific choices. Topic files are created when their first decision is recorded. The root `DECISIONS.md` is an index with one line per decision. Decision numbers (D1, D2, …) are unique across all files. Decisions are never deleted: a changed decision gets a new entry, and the old entry is marked "Superseded by Dn".
**Alternatives considered:** One file with entries tagged by area (simple, but grows long and mixes dataset-specific with harness-wide rules). A `DECISIONS.md` in each code folder (local to the code, but orphaned when folders move, and cross-cutting topics such as splits or evaluation have no home).
**Rationale:** Topic files stay stable while the code structure evolves; global numbering keeps references unambiguous; never deleting entries keeps older results traceable to the rules in force when they were produced.

## 2026-10-03 — D11: One repository; studies in `studies/`; one-way imports
**Decision:** All work lives in this repository: the harness as an installable library, and study-specific code in `studies/<name>/`. Study code imports only the harness's public functions; the harness never imports from studies. Each run records its git commit; the commit used for a paper is marked with a git tag. At publication, the code for that paper is copied into a separate publication repository, and development continues in this repository. Conflicting library versions are handled with separate Python environments, not separate repositories.
**Alternatives considered:** Harness and each study in separate repositories, studies pinning a harness commit (cleaner boundaries, but more overhead for a single researcher). One repository without an import rule (simplest, but study code would depend on harness internals and harness changes could silently alter study results).
**Rationale:** A single repository is simplest for one researcher. The import rule and recorded commits guard against the real risks: harness changes silently altering earlier results, and entanglement that makes the publication copy hard to extract.

## 2026-10-06 — D14: Python environments with uv; Python 3.12
**Decision:** Environments are managed with uv (`pyproject.toml` plus lock file). The harness uses Python 3.12. Wrapped methods that need older Python or library versions get their own small uv environment.
**Alternatives considered:** Poetry (mature and widespread, but does not install Python versions itself and its PyTorch package-index setup is fiddly). conda (handles CUDA, but slower and with clumsier lock files). pip + venv (no lock file).
**Rationale:** Exact reproducible versions, one-command installation of older Python versions for old method repositories, straightforward PyTorch installation, and speed.

## 2026-10-06 — D16: One plain YAML config file per experiment
**Decision:** Each experiment is described by one YAML file, read by a few lines of code with no merging or override machinery. Every parameter is written in the file; nothing is filled in implicitly.
**Alternatives considered:** Hydra (powerful composition and sweeps, but more to learn to read a config). Python dataclasses as configs (type-checked, but less readable).
**Rationale:** Easiest to read and check (code-style rule in CLAUDE.md); no hidden defaults. Can be revisited if configs start to repeat heavily.

## 2026-10-06 — D17: Run results as plain files
**Decision:** Each run writes a folder under `runs/` containing a copy of its config, the git commit, the seed, metrics as JSON/CSV and a log. `runs/` is not committed.
**Alternatives considered:** MLflow (local web interface for comparing runs; can be added later). Weights & Biases (cloud-based; results leave the machine).
**Rationale:** Nothing extra to install; every result can be inspected in a text editor.

## 2026-10-06 — D18: Package layout
**Decision:** Flat layout: `harness/` (library: `data/` with one reader per dataset and `canonical.py`, `windows.py`, `targets.py`, `methods/` with one file per method, `evaluate.py`, `runs.py`), `scripts/` (small entry points), `configs/` (YAML), `external/` (official method repositories, D19), `studies/<name>/` (D11), `tests/`, `runs/` and `data/` (`raw/`, `canonical/`; both not committed). Files and folders are created only when first needed.
**Alternatives considered:** The "src layout" (`src/harness/`): guards against some import mistakes, but adds a folder level with little gain for a single-user project.
**Rationale:** Small, flat and readable; each file has one clear job.

## 2026-10-08 — D33: Representations saved with a window table; overlap check before pretraining
**Decision:** Every run writes `windows.csv` in its run folder, with one row per stored representation: `split`, `row` (row in `Z_<split>.npy`), `run_id`, `capture_id`, `channels`, `start`, `stop` (samples, stop exclusive). The table is the same for all seeds, so it is written once per run. Before pretraining, the run stops with an error if any (run, capture) is in both the training and the test set.
**Alternatives considered:** Rows of `Z` matched to labels by position only (the previous behaviour; any target not saved at pretraining time would need the encoder re-run, or would rely on the loader always returning rows in the same order).
**Rationale:** Step 2 probes the same representations for targets that do not exist yet (band energies of the same window, speed, temperature, operating condition); with the table, each is a lookup through the reader. It also allows error breakdowns per recording or bearing, and enforces in code that no test capture is used for pretraining. The check works at capture level; a stricter unit (recording, bearing) is part of each dataset's split decision.

## 2026-10-12 — D63: Results document in `report/`, numbers generated from the runs, numbered result entries (superseded by D98)
**Decision:** Results are documented in one LaTeX document, `report/results.tex`. Its tables, figures and every number quoted in the prose are generated by `scripts/make_report.py` from the run folders into `report/tables/` and `report/figures/`, which are committed (the run folders are not, D8, D17). Each result is an entry R1, R2, … numbered in run order and never renumbered, with question, test, result, finding, consequence and open points. Status per entry: *Current*; *Superseded by Rn (kept)* when it explains a decision or shows something the later entry does not (moved to an appendix); or *superseded and replaced* for a plain rerun (text removed, number and replacement kept in the overview table). The user decides each entry's status.
**Alternatives considered:** `studies/part1/report/` (D11; ties the document to one study) or `docs/results/`. Numbers typed by hand into the `.tex` (simpler, but can drift from the run files). Only current results (history left to git and the decision log). A chronological lab notebook keeping everything.
**Rationale:** User's decision (2026-10-12). Results that were superseded can still explain decisions; generated numbers cannot drift from the runs.

## 2026-10-14 — D68: Dataset examination code in `eda/<dataset>/`
**Decision:** Code for examining datasets that is not part of any pipeline for testing and comparing methods (EDA notebooks, one-off checks and diagnostics) lives in `eda/<dataset>/`, e.g. `eda/herning/`. It imports only the harness's public functions (as studies do, D11); nothing in `harness/`, `scripts/` or `studies/` imports from `eda/`. Outputs that must be kept go to run folders under `runs/` as before. The existing Herning checks (`check_bands.py`, `spectra_herning.py`, `check_vfd_lines.py`) are moved from `scripts/` to `eda/herning/`; their configs stay in `configs/`. Notebook tools (jupyter, matplotlib, scipy) are a separate uv dependency group `eda`, so the pipeline environment is unchanged. Notebooks are committed with outputs cleared.
**Alternatives considered:** `scripts/eda/<dataset>/` (all runnable code in one place, but blurs pipeline and exploration). `studies/eda/` (consistent with D11, but studies are meant for paper-specific work). Leaving the existing checks in `scripts/` (no references to update). Notebook tools as main dependencies (simpler, larger pipeline environment). Notebooks committed with outputs (plots visible without running, but large, noisy diffs) or not committed at all.
**Rationale:** User's decision (2026-10-14). Keeps `scripts/` for pipeline entry points and gives exploratory work a clear home per dataset.

*Note (2026-10-20): D91–D94 were approved on 2026-10-17 and first recorded as D71–D74 in commit 9655c07, made in parallel with commit 148453c, which used D71–D74 for other decisions. Renumbered when the two were merged; content unchanged apart from the decision numbers.*

## 2026-10-17 — D91: `studies/` holds studies and paper material; shared diagnostics in `harness/diagnostics.py`
**Decision:** `studies/<name>/` (D11) is used for studies as well as material for papers; a specific publication may later get its own repository (as D11 already allows). Diagnostics that do not depend on a method (e.g. the collapse measures of D93) live in `harness/diagnostics.py`, so harness methods and studies can both call them (D11: the harness never imports from studies). The first study is `studies/tfc_ablation/` (D92).
**Alternatives considered:** Ablation stages as switches in `harness/methods/tfc.py` (no new folder, but D41 rejected switches for every fix, and it would change the method behind R6–R7). Everything, diagnostics included, inside the study (self-contained, but other methods such as TS-JEPA could not use the diagnostics without a copy). A separate top-level folder for non-paper studies (not needed: the repository holds both).
**Rationale:** User's decision (2026-10-17). Reusable diagnostics; the validated `tfc.py` stays unchanged.

## 2026-10-20 — D98: Results document reports current results by topic; a generated run index replaces numbered entries
**Decision:** (1) `report/results.tex` reports current results only, organised in topic sections; entries are named by topic, not numbered. (2) As before (D63), every table, figure and quoted number is generated by `scripts/make_report.py` from the run folders, and each entry names its run folders. (3) A generated run index (appendix) lists every folder in `runs/`: what it was for, its status (the section that reports it, superseded by which folder, empty, examination, or not written up yet), its commit (marked when the working tree had uncommitted changes) and device. It is kept in `RUN_INDEX` in `make_report.py`; a folder missing there is listed as "not indexed" and a warning is printed. (4) Superseded results leave the text; where a later result builds on one, it is kept as a sentence. Applied on 2026-10-20: the ten entries regrouped by topic; the TS2Vec and T-Loss FordA entries (and the CPU check) replaced by the reruns with timing, their earlier runs listed as superseded in the index; new FordA comparison of the SSL methods with the baselines (D80); the provenance table replaced by the run index. Supersedes D63.
**Alternatives considered:** D63 as it stood: a complete chronicle, numbered in run order, superseded entries kept in an appendix (cluttered, and numbering in run order needs coordination when work runs in parallel). Results only, without an index (runs not reported would be hard to find).
**Rationale:** User's decision (2026-10-20). The Part 1 report is for the user and the supervisor, so readable current results matter more than chronology; the run index and the decision log keep every result traceable, including failed and superseded ones.

## 2026-10-20 — D99: Runs for the report start from a committed tree; uncommitted changes give a warning
**Decision:** Runs whose results go into the report are made from a committed working tree, so that the commit recorded in `git.json` reproduces the code. `runs.new_run_dir` (used by every script that creates a run folder: `scripts/pretrain.py`, `scripts/probe_spectrum.py`, the TF-C study scripts) prints a clear warning when the working tree has uncommitted changes; the run still starts, and `git.json` records the state as before (shown with an asterisk in the run index, D98). Prompted by the FordA baseline runs of 2026-10-20, which started with uncommitted changes.
**Alternatives considered:** Leave as is (state recorded, no warning). Refuse to start with uncommitted changes unless an explicit, recorded flag is given (safest, but blocks quick test runs, and often while two sessions edit the repository in parallel).
**Rationale:** User's decision (2026-10-20). The warning comes at the moment it matters, with little friction.

## 2026-10-20 — D100: Figures of a study in a notebook in the study folder, committed with outputs cleared
**Decision:** The figures of a study are drawn by a notebook in the study's folder (first: `studies/tfc_ablation/figures_forda.ipynb`), with the plotting functions in a module next to it (`plots.py`) and the run folders and settings in a config (`configs/plots_forda.yaml`). The figures are shown inline only; no figure files are written, and the notebook is committed with outputs cleared (as notebooks in `eda/`, D68), to keep the repository small.
**Alternatives considered:** A script writing PDFs into `studies/<study>/figures/` (figures kept with the study and usable in the report directly; first version, replaced the same day). The notebook also writing the PDFs. The notebook committed with outputs (figures visible without running, but large, noisy diffs).
**Rationale:** User's decision (2026-10-20). Interactive viewing; nothing large in git. Figures for the report are made separately (D98).

## 2026-10-21 — D103: FordA report: TS-JEPA (masking fixed) and PatchTST added; "not timed" rows; TS-JEPA reproduction check on the official-masking run
**Decision:** (1) The FordA comparison in the results document (accuracy, paired and timing tables, figure) includes TS-JEPA (masking fixed, D102; run `20261016_114616_tsjepa_forda_fixed`) and PatchTST (mean pooling, D87; run `20261019_154248_patchtst_forda_mean`), from their existing runs. (2) Neither has stage timing (D79); both appear in the timing table as "not timed" rows. (3) TS-JEPA's reproduction check uses the official-masking run (`20261016_114600_tsjepa_forda`, the paper's code) with the SVM probe (as TS2Vec and T-Loss) against the paper's 0.915; that run appears in the reproduction table only, not in the comparison (D102).
**Alternatives considered:** (2) Leaving both out of the timing table with a note in the caption. (3) The masking-fixed run as the reproduction row; no row, with the explanation in the text only.
**Rationale:** User's decision (2026-10-21), following Claude's recommendations. The missing timing stays visible in the table. The reproduction check asks whether the wrapping of the paper's code works, which is separate from which variant is compared.
