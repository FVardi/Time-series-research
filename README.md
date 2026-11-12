# Time-series research harness

Baselining and developing self-supervised learning methods for time series, with a focus on
condition monitoring and lubrication condition monitoring of ball bearings (PhD, Aarhus University).
Collaboration rules for Claude: `CLAUDE.md`.

## Where to find what

| question | where |
|---|---|
| What has run, what is ready, what is missing? | `STATUS.md` — generated: `uv run python scripts/status.py` (D261) |
| What is still to do? | the project checklist (claude.ai Project doc `claude/CHECKLIST.md`, D258) |
| What are the current rules? | `docs/protocol.md` (all datasets and methods), `docs/methods/`, `docs/datasets/` (D259) |
| Why is it like that, and when was it decided? | `DECISIONS.md` (index) and `decisions/` (full entries, never deleted) |
| Questions for the supervisor | `QUESTIONS.md` |
| Ideas for the own method; candidate datasets | `METHOD_IDEAS.md`; `DATASET_CANDIDATES.md` |
| Results report (updated only when asked) | `report/results.tex` |

## Layout (D18, D262)

| folder | contents |
|---|---|
| `harness/` | the library: `data/` (one reader per dataset, canonical format), `methods/` (SSL methods: adapters and our own), `baselines/` (methods without training), `features/` (stored feature sets: Herning band features, MSSP statistics, sub-band power), `windows.py`, `labels.py`, `protocols.py`, `evaluate.py` (probes, metrics), `runs.py` (configs, run folders), `diagnostics.py` |
| `scripts/` | entry points: `pretrain.py` + `evaluate.py` (one run), `run_combinations.py` (Paderborn), `compare.py`, `status.py`, `convert_<dataset>.py`, `derive_features.py`, queue scripts |
| `configs/` | one file per standard run or variant, `<method>_<dataset>[_variant].yaml`; shared parts in `configs/datasets/` and `configs/methods/`, merged through `defaults:` (D260); `configs/compare/` (D203); `configs/synthetic/` |
| `studies/` | one folder per study (`studies/README.md`) |
| `eda/` | examination of datasets, with their configs (D68) |
| `tests/` | `uv run pytest tests` |
| `external/` | official method code, pinned submodules, never edited (D19) |
| `data/`, `runs/` | not committed (D8, D17); superseded runs in `runs/archive/` (D239) |

## Running

    uv sync
    uv run python scripts/pretrain.py configs/<method>_<dataset>.yaml --seeds 0
    uv run python scripts/evaluate.py runs/<run folder>
    uv run python scripts/status.py
