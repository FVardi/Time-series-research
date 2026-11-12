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

## 2026-10-06 — D14: Python environments with uv; Python 3.12 (Python version superseded by D132)
**Decision:** Environments are managed with uv (`pyproject.toml` plus lock file). The harness uses Python 3.12. Wrapped methods that need older Python or library versions get their own small uv environment.
**Alternatives considered:** Poetry (mature and widespread, but does not install Python versions itself and its PyTorch package-index setup is fiddly). conda (handles CUDA, but slower and with clumsier lock files). pip + venv (no lock file).
**Rationale:** Exact reproducible versions, one-command installation of older Python versions for old method repositories, straightforward PyTorch installation, and speed.

## 2026-10-06 — D16: One plain YAML config file per experiment (amended by D260)
**Decision:** Each experiment is described by one YAML file, read by a few lines of code with no merging or override machinery. Every parameter is written in the file; nothing is filled in implicitly.
**Alternatives considered:** Hydra (powerful composition and sweeps, but more to learn to read a config). Python dataclasses as configs (type-checked, but less readable).
**Rationale:** Easiest to read and check (code-style rule in CLAUDE.md); no hidden defaults. Can be revisited if configs start to repeat heavily.

## 2026-10-06 — D17: Run results as plain files
**Decision:** Each run writes a folder under `runs/` containing a copy of its config, the git commit, the seed, metrics as JSON/CSV and a log. `runs/` is not committed.
**Alternatives considered:** MLflow (local web interface for comparing runs; can be added later). Weights & Biases (cloud-based; results leave the machine).
**Rationale:** Nothing extra to install; every result can be inspected in a text editor.

## 2026-10-06 — D18: Package layout (extended by D262)
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

## 2026-10-27 — D130: STATUS.md: living status list at the repository root (superseded by D261)
**Decision:** `STATUS.md` at the repository root lists what is done, ready to run and open (benchmark runs per dataset, methods, datasets, loose ends, deferred items) and is updated as things change. Decisions stay in `decisions/`, supervisor questions in `QUESTIONS.md`. It takes over the role of `HANDOFF.md`, which dates from early October and no longer describes the repository (whether to delete it is up to the user).
**Alternatives considered:** Rewriting `HANDOFF.md`; a section in `README` or `CLAUDE.md`.
**Rationale:** User's request (2026-10-27).

## 2026-10-27 — D132: Python 3.14 for the harness (supersedes the Python version in D14)
**Decision:** The harness uses Python 3.14, the version its environment actually runs (`.venv/pyvenv.cfg`: CPython 3.14.3). The rest of D14 (uv, `pyproject.toml` + lock file, separate small environments for methods that need other versions) stands. `pyproject.toml` keeps `requires-python = ">=3.12"`.
**Alternatives considered:** Pinning 3.12 as D14 said (`uv python pin 3.12`, `uv sync`); the existing runs were made on 3.14, so they would no longer match the environment.
**Rationale:** User's decision (2026-10-27): no reason to go back; all runs so far on this machine used 3.14.

## 2026-10-28 — D147: Windowed inputs with the split set in the config (`harness/windows.py`, `harness/labels.py`)
**Decision:** Datasets whose split is not an official one (first CWRU) get their inputs through two new modules: `harness/windows.py` (generic: select captures by filters on the captures table, with the stream's sample rate joined; load, resample to one rate if needed and cut consecutive windows; per-channel standardisation; check that no test capture or test group is in the pretraining data) and `harness/labels.py` (one function per dataset deriving class and group from the captures table; label derivation stays out of the adapters, D4). `scripts/pretrain.py` takes this path when the config has a `split` section (FordA and FD unchanged): pretraining on the `train` windows plus any `pretrain_only` windows, probes on `train` and `test`; the window table gets `label` and `group` columns. `scripts/evaluate.py` uses the grouped folds of D129 when the config sets `evaluation.cv_groups: true`. Config format:
```yaml
labels: cwru
windows: {length: 2048, sample_rate_hz: 12000, standardise: true}
split:
  train: {sample_rate_hz: 12000, fault_end: [DE, FE], fault_size_mil: [7, 14]}
  test: {sample_rate_hz: 12000, fault_end: [DE, FE], fault_size_mil: [21]}
  pretrain_only: {fault_end: [none]}
evaluation: {probes: [logreg, floor], cv_groups: true, ...}
```
**Alternatives considered:** A CWRU-only loader (quicker, but Paderborn and Herning would need their own).
**Rationale:** User's decision (2026-10-27/28). The same windowing, split selection and checks serve every dataset without an official split.

## 2026-10-29 — D162: Submodules external/tstcc and external/frera ignore untracked files
**Decision:** In `.gitmodules`, `external/tstcc` and `external/frera` get `ignore = untracked`, as `external/tloss` and `external/minirocket` already have. Their only untracked content is Python's `__pycache__/` folders, which made `git status` report the tree as changed, so every run recorded `uncommitted_changes: true` (D99) even right after a commit. Changes to tracked files in these submodules are still reported.
**Alternatives considered:** `ignore = dirty` (also hides edits to the official code, which would make git.json miss changes that affect results). Leaving it as it was.
**Rationale:** User's decision (2026-10-29). The record of uncommitted changes (D99) should mean something; the narrower setting keeps real code changes visible.

## 2026-11-05 — D237: FreRA and TF-JEPA save only the weights encode() uses
**Decision:** `harness/methods/frera.py` saves the FCN encoder (`encoder_q` without its `logits` projector) and the FreRA frequency weights; `harness/methods/tfjepa.py` saves the online encoders and projectors (not the EMA targets and predictors). Representations, probes and results are unchanged; nothing in the harness loads these files. Earlier runs keep their full `model.pt`.
**Alternatives considered:** Keep saving the whole model (FreRA on CWRU: 4.4 GB per seed, from the official `Linear(d, d)` projector on the 33,024-dimensional flattened FCN output; TF-JEPA on CWRU about 1 GB per seed). Not saving models at all (the encoder could then not be probed again later).
**Rationale:** User's decision (2026-11-05), repo cleanup: disk space; the saved parts are all that is needed to encode again.

## 2026-11-05 — D238: Text files stored with LF line endings (`.gitattributes`)
**Decision:** `.gitattributes` with `* text=auto eol=lf`: git stores text files with LF, whatever line endings an editor or tool writes on Windows; binary files are detected and left alone.
**Alternatives considered:** No rule (files rewritten with CRLF show as completely changed; about 60 such files on 2026-11-05). `core.autocrlf` per machine (not stored in the repo, so each computer must be set up).
**Rationale:** User's decision (2026-11-05), repo cleanup: diffs and commits show only real changes.

## 2026-11-12 — D258: Every chat keeps the checklist up to date itself
**Decision:** The project to-do list (claude.ai Project doc `claude/CHECKLIST.md`) is updated by the chat that does the work: when a chat finishes a task, starts or finishes runs, records a decision or hits a new open problem, it reads the checklist, changes only the lines about its own work and writes the whole doc back (re-reading and redoing the edit if another chat changed it meanwhile). Rule in `CLAUDE.md`, in the checklist's header and (by the user) in the Project instructions. The dedicated checklist chat tidies up and catches what was missed.
**Alternatives considered:** Only the checklist chat edits the checklist, with other chats reporting to the user (the setup since 2026-11-04; the user has to relay every change by hand).
**Rationale:** User's decision (2026-11-12): relaying every change to the checklist chat was cumbersome.

*Note (2026-11-12): the dedicated checklist chat is retired (user); the chats that do the work keep the checklist up to date, and any chat may tidy it.*

## 2026-11-12 — D257: Standard choices without asking; the user decides only what has no standard
**Decision:** Where a choice has a standard, Claude takes it without asking, in this order: an existing rule of the harness (`docs/protocol.md`, the `docs/` cards), the paper or official code or the dataset authors' protocol, then common practice in the field. Claude asks only when there is no standard, sources disagree, a choice would depart from a standard, or it is costly or cannot be undone. D-entries are written for the user's decisions and for departures from a standard; standard choices are recorded in the configs and the cards, with their source. Rule in `CLAUDE.md`.
**Alternatives considered:** The user decides every choice and Claude asks each time (the rule until now: many questions, slow progress, many branches of decisions).
**Rationale:** User's decision (2026-11-12): asking every time creates bloat and slows progress; industry and literature standards are used wherever they exist.

## 2026-11-12 — D259: Current rules in `docs/` cards; the decision log stays the history
**Decision:** The rules in force are kept in `docs/`: `docs/protocol.md` (rules for every dataset and method, each citing its D-entries), `docs/methods/<method>.md` and `docs/datasets/<dataset>.md` (one card each: what is used, settings, departures from the source, status). `docs/README.md` is the index and says when a card is updated (with the decision that changes it). `decisions/` stays the dated, never-deleted record of why.
**Alternatives considered:** Reading the current rule out of the log (a rule is often spread over several entries that amend each other).
**Rationale:** User's request (2026-11-12), applying the restructuring proposed on 2026-11-03: what is in force can be found in one place.

## 2026-11-12 — D260: Configs composed from dataset and method parts (amends D16)
**Decision:** Settings shared by every config of a dataset are in `configs/datasets/<dataset>.yaml`, settings shared by every config of a method in `configs/methods/<method>.yaml`. A config lists its parts under `defaults:` (paths relative to the file); `harness/runs.py:load_config` merges them in order (mappings key by key, everything else replaced) and the file's own keys last. The run folder gets the merged config as `config.yaml` and copies of the source files in `config_sources/`. All configs were checked to resolve to exactly the settings they had before. `configs/tsjepa_paderborn.yaml` stays as it is until the running GPU queue has finished.
**Alternatives considered:** One complete file per experiment (D16; the same dataset and method settings repeated in up to 137 files). Hydra (more machinery than needed).
**Rationale:** User's request (2026-11-12), applying the restructuring proposed on 2026-11-03; D16 anticipated revisiting it if configs repeat heavily. Every setting is still written in a file, and the run folder still holds the full config.

## 2026-11-12 — D261: `STATUS.md` generated by `scripts/status.py` (supersedes D130)
**Decision:** `STATUS.md` is written by `scripts/status.py` from the configs, the run folders, the queue logs, `docs/datasets/` and the `Status:` line of each study README; it is not edited by hand. Run it after runs finish (`uv run python scripts/status.py`). What is still to do stays in the project checklist (D258).
**Alternatives considered:** The hand-kept `STATUS.md` of D130 (it fell behind the runs).
**Rationale:** User's request (2026-11-12), applying the restructuring proposed on 2026-11-03: keeping track of what is done was the hardest part.

## 2026-11-12 — D262: Layout: baselines and feature sets in their own packages; study and EDA scripts with their study (extends D18, D68, D91)
**Decision:** Methods without training (`raw`, `fft`, `envelope`, `catch22`, `minirocket`, `mssp`, `mssp_features`, `pooled`) moved from `harness/methods/` to `harness/baselines/`; the stored feature sets (`band_features`, `subband_power`, `mssp_stats`, `stored_features`) from `harness/` to `harness/features/`. The TF-C reproduction scripts and configs moved to `studies/tfc_reproduction/`, two one-off method checks to `studies/method_checks/`, and EDA configs to `eda/<dataset>/configs/`. References in code, configs, notebooks and `docs/` were updated; `decisions/` and `report/` are left as written. Each study README starts with a `Status:` line; `studies/README.md` and a root `README.md` index the repository.
**Alternatives considered:** Keeping the layout (`harness/methods/` mixing SSL methods with baselines; study scripts among the entry points in `scripts/`).
**Rationale:** User's request (2026-11-12), applying the restructuring proposed on 2026-11-03.

## 2026-11-12 — D263: Experiments start simple and grow step by step; every extension applies to all methods or none
**Decision:** (1) *Simple first, then build.* An experiment or study starts in its simplest form that answers its question; refinements (extra measures, reweightings, extra targets, finer settings) are added one at a time, only after the simple version has run and been looked at, and only when the user asks for them. Proposed refinements are noted as options (study README or checklist), not built. (2) *Methods treated alike.* Any extra beyond the standard runs (longer training, extra checkpoints, more seeds, extra variants or diagnostics) is done for every method in the comparison or for none; one method is not given a special case. Existing one-method diagnostics (e.g. the FordA longer-training runs of D142) stay as the record but are not used as extra points in comparisons. Rule in `CLAUDE.md` and `docs/protocol.md` (P28).
**Alternatives considered:** Adding refinements as they come up (e.g. relevance-weighted R² in the spectral probe, D245 (b)); using the D142 longer-training runs as extra points for TF-JEPA, SimMTM and TF-C only.
**Rationale:** User's request (2026-11-12): keep experiments simple and then build; either all methods get more detail or none.
