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
