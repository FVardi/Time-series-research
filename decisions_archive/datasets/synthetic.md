# Synthetic data — decisions

## 2026-11-03 — D196: Synthetic test bed with known factors (METHOD_IDEAS M5)
**Decision:**
1. *Integration:* canonical datasets written once by `scripts/make_synthetic.py` from a config (`configs/synthetic/*.yaml`); generator in `harness/data/synthetic.py`; every method and config runs through the harness unchanged. Labels in `harness/labels.py`: `synthetic` (class = state, group = bearing) and `synthetic_kappa` (kappa_like, regression, group = bearing). All factors stored on the captures table.
2. *Signal model (toy, not lubrication physics):* speed -> shaft harmonics (amplitude 1/k) plus one characteristic order; lubrication state -> Poisson bursts of decaying sines (rate and amplitude per state); temperature -> broadband noise level; bearing -> gain and resonance filter, drawn once per bearing and sensor.
3. *Size:* 4 speeds x 3 temperatures x 4 bearings, 144 runs per design, 32,768 samples per run (32 windows of 1,024) at a nominal 10 kHz; values stored as generated, float32.
4. *Designs:* crossed (every combination), confounded (state = class of kappa_like(speed, temperature)), departures (as confounded, a third of the runs with another random state); three datasets.
5. *Effect sizes:* operating point strong, state subtle by default; changeable in the config.
6. *Two sensors now:* AE-like (bursts dominate) and US-like (periodic part dominates), sharing speed, temperature, state and burst times; carrier, weights and bearing nuisance per sensor.
7. *Targets:* class = state; group = bearing; kappa_like stored for every run.
**Alternatives considered:** an in-memory generator for study scripts only; one sensor; one design.
**Rationale:** User's decision (2026-11-03), approving Claude's proposal. FordA has no condition labels; the toy gives exact targets for the factor x block probes (D184). Numbers in the configs are Claude's choices within the approved design.
