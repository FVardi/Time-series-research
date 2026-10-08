# Decisions — FD-A / FD-B (TF-C's Paderborn windows)

Dataset-specific decisions for FD-A and FD-B. Index: `../../DECISIONS.md`.

## 2026-10-08 — D39: FD-A and FD-B used as distributed by the TF-C authors
**Decision:** FD-A and FD-B are converted exactly as distributed on figshare (articles 19930205 and 19930226): every window is one capture, all three files (train, val, test) are converted, values stored as they are (64-bit floats, scale 1, offset 0, units unknown). Dataset `fd`, one run per operating condition (`fd_a`, `fd_b`). Class label (0, 1, 2) and official split (`train`/`val`/`test`) are extra columns on the captures table (as D30). Status `final`. The files do not say which recording or bearing each window comes from, so leakage between the splits (overlapping windows, shared bearings) cannot be checked; D33's check only confirms that no window is reused. Step 2 of D35 (raw Paderborn data) addresses this.
**Alternatives considered:** Rebuilding the windows from the raw Paderborn data (no longer comparable with the published numbers).
**Rationale:** The reproduction must use the data behind the published result.
