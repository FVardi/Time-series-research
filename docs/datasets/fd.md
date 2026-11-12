# FD-A / FD-B (TF-C's Paderborn windows)

**Source:** TF-C authors' figshare articles 19930205 and 19930226 (windows cut from Paderborn
recordings, 5,120 samples, 3 classes). Converted as distributed: one capture per window, official
train/val/test as a column; dataset `fd`, runs `fd_a`, `fd_b` (D39).
**Role:** record of the TF-C reproduction only, not a benchmark (D70): FD-B is solved by the plain
spectrum (0.999, D46) and leakage between its splits cannot be checked (no recording or bearing
IDs, D39).

## Current rules
- Used only by `studies/tfc_reproduction/` (official TF-C, our TF-C, probe diagnostics).
- Open: FD-A → FD-B transfer with the four TF-C variants (D211): pipeline, probes, seeds.

History: D35–D46, D70, D211.
