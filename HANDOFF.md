# HANDOFF — Time series research harness

## GOAL
Fløderik (PhD, Aarhus University; lubrication condition monitoring of ball bearings with acoustic emission (AE) and ultrasound + ML) is building a research harness to baseline and develop self-supervised learning (SSL) methods for time series. It follows Step 1 of his research plan (claude.ai artifact "Frederik Research Plan"): leakage-safe evaluation of existing SSL methods, first on public data with published numbers, later on his Herning rig data.

## CONTEXT
- Repo: `C:\Users\au808956\Documents\Repos\Time-series-research` on his **work computer** (GitHub remote `FVardi/Time-series-research`, branch `main`). He sometimes chats from a home computer, where the repo/data are not available; runs happen on the work computer (Windows, PowerShell, NVIDIA GPU, CUDA 13.0 driver).
- Stack (decided): Python 3.12 via **uv** (`pyproject.toml` + `uv.lock`), plain PyTorch 2.14.1+cu130 (from `download.pytorch.org/whl/cu130`), scikit-learn, h5py, pandas/pyarrow, pyyaml, pytest. `harness` is an installable package.
- Claude's access: the linked work computer's isolated Linux VM (`device_bash`; often fails to start → fall back to `device_stage_files`/`device_commit_files`), plus a cloud workspace. Neither can install GPU PyTorch; the VM disk is too small for PyTorch at all. Network allowlist: GitHub, PyPI, www.timeseriesclassification.com allowed; download.pytorch.org, figshare, huggingface, zenodo, cs.ucr.edu blocked (user can add domains in Claude settings → Capabilities).
- Layout:
  - `CLAUDE.md` — collaboration + code-style rules (read it first)
  - `DECISIONS.md` — index of decisions D1–D32 (+ D33, see open issues); full entries in `decisions/{data,repo,training,evaluation}.md`, `decisions/datasets/{herning,forda}.md`
  - `harness/data/{canonical,herning,forda}.py` — canonical format + readers
  - `harness/methods/{ts2vec,tloss,external}.py` — wrapped official methods; `external.py` imports official repos with name isolation
  - `harness/evaluate.py` (probes, label subsets, metrics), `harness/runs.py` (config, seeds, run dirs, git commit)
  - `scripts/{convert_herning,convert_forda,pretrain,evaluate}.py`
  - `configs/{ts2vec_forda,tloss_forda}.yaml`
  - `external/ts2vec` (submodule @ b0088e1), `external/tloss` (submodule @ 4aff592) — never edited
  - `tests/test_herning.py`, `tests/test_slice.py`
  - `data/` and `runs/` git-ignored. Data: `data/raw/herning/scope_20260901_112732.h5` (38.8 GB), `data/raw/forda/` (FordA .txt from timeseriesclassification.com), `data/canonical/forda/forda/` (converted). Herning NOT yet converted in full.

## DECISIONS MADE (all user-approved; details in decision files)
- D1–D7, D20–D25 data layer: run → capture → stream → channel; four metadata tables (runs, captures, channels, telemetry); canonical = Parquet tables + one HDF5 per run; values stored as recorded (int16 codes + scale/offset for the 8-bit scope), no ingestion transforms; run status field (`validation`/`provisional`/`final`); HDF5 chunk 16,384, gzip 4; `load_signal` returns float64, float32 conversion at training; source identified by name/size/mtime (D24, replaces checksum part of D5); Herning: only AE + UL channels converted for now (D25), status `validation` (rig unfinished).
- D8 `/data/` ignored; D9 decision log split by topic, global numbering, never delete (mark "Superseded by Dn"); D11 one repo, study code in `studies/<name>/`, one-way imports, tag per paper (publication = copy to separate repo).
- D10 methods: (A) wrapped official code, (B) composed from harness components; both expose an encoder; composed versions must reproduce official ones first.
- D12 κ (Herning regression target) computed exactly as in the user's MSSP paper; open details listed in the entry.
- D13 tests chosen per research question and dataset; full grid within a study; built incrementally (only necessary comparisons).
- D14–D19 stack/layout (above); D19 official repos as pinned submodules, fixes in adapters, modified versions under their own names.
- D26/D29 evaluation (first slice): frozen encoder; probes = TS2Vec's SVM protocol (`svm_ts2vec`, official code) + harness logistic regression (standardised, C by 5-fold CV over 1e-4..1e4, accuracy); test set used once; accuracy + macro-F1; 3 seeds; label budgets 1/10/100 % (new stratified subset per seed); reproduction criterion: published value within mean ± 2 SD, or within 1 pp if spread tiny.
- D27/D30/D31 FordA as distributed (already z-normalised by archive), labels/official split as extra captures columns, status `final`.
- D28 TS2Vec official defaults (lr 0.001, batch 8, 320 dims, 600 iters); D32 T-Loss official settings with 2,000 steps (paper, K=10), target 0.928.
- Deferred by user: splits, input representation for long Herning sweeps, statistics (CD diagrams etc.), fine-tuning protocol.

## CURRENT STATE
- *Working / verified:* Herning conversion code (tested on 20 sweeps: exact); FordA converted (exact vs files). TS2Vec on FordA (GPU, commit adf16c6): SVM 100 % 0.932 ± 0.002 vs published 0.936 → PASS (via the 1 pp clause). T-Loss on FordA (GPU): 0.933 ± 0.005 vs 0.928 → PASS. Both methods indistinguishable on FordA. Run folders: `runs/20261008_084722_ts2vec_forda`, `runs/20261008_092321_tloss_forda`, plus `runs/20261007_214220_ts2vec_forda_cpu_check` (cloud CPU check).
- *Uncommitted (probably):* T-Loss adapter/config/submodule, `harness/methods/external.py`, `scripts/pretrain.py` "Next:" line. User was asked to commit.
- *Not done:* full Herning conversion (~5 min; `uv run python scripts/convert_herning.py data/raw/herning/scope_20260901_112732.h5 data/canonical`).

## KEY ARTIFACTS
Run commands (PowerShell, repo root):
```
uv sync
uv run pytest
uv run python scripts/convert_forda.py data/raw/forda data/canonical
uv run python scripts/pretrain.py configs/<method>_forda.yaml   # prints "Next: ..." evaluate command
uv run python scripts/evaluate.py runs/<run folder>            # writes results.csv, summary.csv, prints reproduction check
```
Adapter interface every method module provides: `pretrain(X, settings, device) -> (model, loss_log)`, `encode(model, X) -> (n, d)`, `save(model, path)`; X is (series, time, channels) float64. Register in `METHODS` in `scripts/pretrain.py`.
Canonical layout: `<root>/<dataset>/<run_id>/{runs,captures,channels,telemetry}.parquet + signals.h5` with datasets `/<capture_id>/<channel>` carrying attrs scale/offset/units.

## DEAD ENDS
- Running `git status` normally from the VM leaves `.git/index.lock` (no delete rights) → use `git --no-optional-locks`; git writes need delete permission (`device_request_delete_permission`).
- PyTorch in the VM: disk too small / CUDA libs missing → test in cloud workspace (venv `/tmp/claude-0/venv312`, may be gone).
- Placeholders like `<the new folder>` in commands: user typed them literally (PowerShell `<` error) → always give exact paths.
- T-Loss `losses/__init__.py`/`networks/__init__.py` break on Python 3.12 → stubbed in adapter via `import_official(..., stub_packages, preload)`. Its `joblib`/`iid` code is avoided by not using its classifier.

## OPEN ISSUES
- `scripts/pretrain.py` contains changes Claude did not make (writes `windows.csv`, calls `runs.check_no_overlap`, cites "D33"). Origin unknown (user or another session); verify D33 exists in DECISIONS.md and that `runs.check_no_overlap` exists before running.
- κ details need from user: MSSP κ code/formula, oil viscosity data, bearing mean diameter; then decide speed/temperature source, window assignment, exclusions, target form (κ vs log κ vs classes). Telemetry is read a few seconds before each sweep.
- Herning: SP channel meaning unknown; `test_parameters` metadata contradicts actual schedule; sample rate 2.5 MHz in this run (older MSSP data 12.5 MHz; final rate unknown).
- scikit-learn warns `probability` param in SVC deprecated (TS2Vec's code); fine while `uv.lock` pins sklearn < 1.11.
- Empty folder `data/AU Herning test rig/` left behind.

## NEXT STEPS
1. Confirm the user committed; check the D33/`windows.csv` question.
2. Propose TF-C on FD-A → FD-B (Claude recommended; user hasn't approved yet): adds transfer + fine-tuning; needs decisions on FD-A/FD-B used as distributed (pre-windowed, TF-C split), fine-tuning definition, TF-C settings, reproduction target; data on figshare (user must allowlist it). Then possibly PatchTST/TS-JEPA, then Herning.
3. When κ inputs arrive: Herning target derivation and input-representation proposals.

## WORKING PREFERENCES
- User makes all design decisions; Claude proposes numbered options (D-style) with trade-offs + recommendation, waits for approval, records every approved decision in the topic file + index. No silent defaults; mark placeholders.
- Code minimal, readable, commented; no frameworks/registries; mention mechanical changes.
- Explain unfamiliar terms plainly; concise answers; incremental work — only necessary comparisons.
- Don't commit or push unless asked; give exact copy-paste PowerShell commands.
