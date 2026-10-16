# eda/ — examining datasets (D68)

Code for looking at datasets that is **not** part of the pipeline for testing and
comparing methods: exploratory notebooks, one-off checks and diagnostics.
One folder per dataset (`eda/herning/`, …).

Rules
- Import only the harness's public functions; nothing outside `eda/` imports from here.
- Results that must be kept (because a decision or a report entry refers to them) are
  written to a run folder under `runs/`, as everywhere else.
- Notebooks are committed with outputs cleared (VS Code: "Clear All Outputs";
  Jupyter: Edit → Clear Outputs of All Cells).

Environment: notebook tools are in the uv group `eda` (not installed by default):

    uv sync --group eda

Contents
- `herning/herning_eda.ipynb` — interactive examination of the Herning source file
  (structure, run schedule, raw sweeps, spectra, envelopes, SP channel, statistics
  over the run, downsampling).
- `herning/check_bands.py` — D52 band check (moved from `scripts/`).
- `herning/spectra_herning.py` — average spectra per operating point (moved from `scripts/`).
- `herning/check_vfd_lines.py` — frequency-converter line comb test, QUESTIONS.md Q3 (moved from `scripts/`).
  The three scripts are run from the repo root, e.g.
  `uv run python eda/herning/check_bands.py configs/band_check_herning.yaml`.
