# Herning test rig (AU; the MSSP rig data) — own data, validation run

**Source:** `data/raw/herning/scope_20260901_112732.h5`: one rig run, one bearing, one lubricant,
rpm swept at 13 temperature targets; AE and ultrasound (UL) at 2.5 MHz as 16-bit codes (D20,
D22, D25). Status `validation` (rig unfinished; final data pending). Not bound by the one-split
rule (D214).
**Config part:** `configs/datasets/herning.yaml`.

## Target
κ = ν(T)/ν₁(n, d_pw) as the MSSP code: rpm from the VFD command × 59.5, temperature
`telem_omron_pv_c`, ASTM D341 viscosity, ISO 281 ν₁, d_pw 38 mm; exclusions as MSSP (rpm target
100, temperature target 25, rpm outside 50–3,600, T < 38 °C). Regression on κ itself (D12, D55).

## Inputs (D165)
Per sweep, a sequence of per-window band features: 12 octave bands (AE 50–100, 100–200, 200–400,
400–560 kHz; UL eight base-10 octaves 1.0–126 kHz), order-5 zero-phase Butterworth, 1 ms windows
(200 per sweep), 4 features (log energy, Hjorth mobility and complexity, kurtosis) → 200 × 48;
standardised per channel. Stored once in `data/derived/herning/band_features.h5` (D172).
1/12-octave sub-band power stored as well, not used yet (D186).

## Evaluation (validation only)
- Split: the sweeps with κ, random 80/20, one fixed draw (D169) — measures recognising the
  operating point, so results weigh less (QUESTIONS.md Q1).
- `ridge` (ungrouped folds, R²) and floor_mean; MAE and R²; label budgets 1/10/100 % (D170).
- Baselines: MSSP statistics (D181), pooled band features (D187); no catch24 (D188).
- Methods: official settings, shape-forced changes only (D173, D195).

## Open
- Intended test: temperature hold-out (D56; QUESTIONS.md Q1); probe folds grouped by hold (D59).
- Sweep vs window representations (Q2); VFD interference and the UL band above 40 kHz (Q3).
- Work resumes when the dataset is updated; MSSP rerun under D255 in the meantime.

History: D12, D20, D22, D25, D52, D55, D56, D59–D61, D165, D169, D170, D172, D173, D181, D186–D188, D195.
Superseded: D51, D62, D64.
