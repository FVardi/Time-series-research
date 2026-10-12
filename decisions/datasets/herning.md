# Decisions — Herning test rig

Dataset-specific decisions for the AU Herning test rig (the MSSP rig data). Index: `../../DECISIONS.md`.

## 2026-10-03 — D12: κ computed as in the MSSP paper
**Decision:** The regression target κ is computed from measured speed and temperature using exactly the computation from the MSSP paper (viscosity–temperature relation, required-viscosity formula and constants). Its inputs and constants (lubricant reference viscosities, bearing mean diameter) are stored in run metadata, not hard-coded. Anything the MSSP computation does not settle is decided separately: which speed and temperature signals are used, how per-sweep telemetry is assigned to windows, which sweeps are excluded (e.g. 0 rpm, unsettled temperature after set-point changes), and the target form (κ, log κ or lubrication regimes).
**Alternatives considered:** A new κ formulation for the harness (free to improve, but results would no longer be comparable with the MSSP paper).
**Rationale:** Keeps harness results directly comparable with published work and avoids silently changing the target definition.

## 2026-10-07 — D22: Herning conversion details
**Status:** Item 1, and the SP/microphone parts of items 4–5, superseded by D25 (2026-10-07).
**Decision:**
1. *Included data:* all scope sweeps (channels AE, UL, SP; one stream at the scope's sample rate) and all microphone captures (ambient and machine microphone; one stream at 80 kHz).
2. *Run:* ID `herning_20260901_112732`, status `validation` (D7).
3. *Capture IDs:* captures sorted by time and numbered with zero padding (`sweep_0000`…, `mic_000`…); the original source names are kept in a column. IDs are labels only; stratification uses metadata columns via the runs, captures and telemetry tables.
4. *Capture time:* the raw timing fields are stored as recorded (`tick`, UTC timestamp, scope trigger offset `x_origin`), plus a derived `start_s` = tick + x_origin used for ordering and stream matching. Timing accuracy: scope "host clock; exact trigger time unknown"; microphones "sub-second" (as stated in the source file).
5. *Sensor types:* AE = `acoustic_emission`, UL = `ultrasound`, SP = `unknown`, microphones = `microphone` (ambient / machine).
6. *Telemetry:* one row per capture with every `telem_*` field as recorded; nothing filtered or derived at conversion.
7. *Run metadata:* bearing, lubricant, scope-settings and test-parameter fields stored in the run's `extras` as recorded, with a note that `test_parameters` contradicts the actual schedule (60 min / 500 rpm / 60 °C vs 13 h, 0–3,000 rpm, 25–100 °C).
**Alternatives considered:** Keeping source capture names (do not sort correctly as text). Excluding the microphones (the harness must handle multimodal data). Defining a single derived capture time only (the relation between `tick` and the trigger is uncertain; raw fields allow revising the definition without reconversion).
**Rationale:** Faithful, complete conversion (D5) with metadata that supports filtering and stratification.

## 2026-10-07 — D25: Only AE and ultrasound converted for now
**Decision:** The Herning conversion includes only the acoustic emission (AE) and ultrasound (UL) channels of the scope sweeps, plus all telemetry and run metadata. The SP channel and the microphone captures are left out for now; they can be added later by reconverting, since the source files are kept unchanged. Supersedes item 1 of D22 and the parts of items 4–5 that concern SP and the microphones.
**Alternatives considered:** Converting all channels and streams (D22 as first approved); raised the unresolved question of how to treat the two microphones, whose sample counts differ (74,153 vs 74,752 at a nominal 80 kHz).
**Rationale:** Only AE and ultrasound are relevant to the current research; converting less keeps the first code smaller and defers questions that do not matter yet.

## 2026-10-09 — D51: Rig input representation: band-limit and downsample, cut-offs after checking where the information is
**Decision:** For the SSL methods, the AE and US signals are band-limited and downsampled per sensor (same input for every method). The cut-off frequency per sensor is decided only after checking that the bands above it carry no information (D52). The user expects useful US content up to about 100 kHz (the MSSP config assumed 40 kHz).
**Alternatives considered:** Short raw windows at full rate (2.5 MHz: windows of a few thousand samples cover 1–2 ms, a fraction of a revolution; attention-based methods cannot take longer windows). AE envelope; frame-wise band energies (both are already feature choices).
**Rationale:** User's decision (2026-10-09). Removes noise bands and buys context without losing information, provided the check confirms it.

## 2026-10-09 — D55: Herning operating point, exclusions and target, as in the MSSP code
**Decision:** Taken from the MSSP code (`publication-ceramicspeed-dataset`, `src/ceramicspeed/loading.py`, `eda.py`, `calculate_kappa.py`, `config.yaml`): rpm reconstructed from the frequency-converter command (`telem_vfd_cmd_hz` × 59.5; the measured rpm is unreliable); temperature from `telem_omron_pv_c`; sweeps excluded with rpm target 100 (bearing not rotating), temperature target 25 (cool-down tail), rpm outside 50–3600, temperature below 38 °C. κ = ν(T)/ν₁(n, d_pw): ν(T) by ASTM D341 (Walther) from the lubricant's 40 °C and 100 °C viscosities in the HDF5 metadata (MSSP fallback 22.0 and 4.1 cSt), ν₁ by ISO 281, d_pw = 38 mm. Regression target: κ itself (not log κ). No classification task for now. Settles the corresponding open details of D12.
**Alternatives considered:** Measured rpm (unreliable). log κ. κ regimes as classes (κ < 0.5, 0.5–1, ≥ 1 in the MSSP config).
**Rationale:** User's decision (2026-10-09).

## 2026-10-09 — D56: Herning split: temperature hold-out (details open)
**Decision:** The Herning data are split by holding out temperature (a block of sweeps at a temperature not seen in training). Open: which block(s) form the test set, whether a validation block is used, and the grouping for cross-validation inside the training set.
**Alternatives considered:** Split by acquisition hold groups (MSSP `grouping.py`).
**Rationale:** User's decision (2026-10-09). Tests generalisation to an unseen operating condition (research plan, Step 1).

## 2026-10-09 — D52: Check where the information is, per frequency band (Herning)
**Decision:** Before choosing downsampling cut-offs (D51) and the broadband upper limits, a one-off check on the Herning run computes, per sensor (AE, US) and per sweep, the power in fine frequency bands up to Nyquist, and reports per band: (1) the ratio of band power in rotating sweeps to that in the sweeps where the bearing did not rotate (rpm target 100; noise reference), and (2) the Spearman correlation of band power with rpm, temperature and κ across the analysed sweeps (D55). Output: a figure and a table per sensor; the cut-offs are chosen by the user from them (no thresholds fixed in advance).
**Alternatives considered:** Taking the MSSP config's ranges (AE 50–600 kHz, US 0–40 kHz, from spectrogram inspection) without a check.
**Rationale:** User's decision (2026-10-09). Downsampling must not remove information; the user expects US content up to about 100 kHz.

## 2026-10-10 — D62: Downsampling factors for the rig signals (US factor superseded by D64)
**Decision:** For the SSL input (D51): AE downsampled by 2 (2.5 MHz → 1.25 MHz, Nyquist 625 kHz, above the 560 kHz signal limit); US downsampled by 5 (2.5 MHz → 500 kHz, Nyquist 250 kHz, above the 178 kHz top of the US bands). Open: anti-alias filter, where the step sits in the pipeline (on the fly or a stored derived copy), and how AE and US are combined when their rates differ.
**Alternatives considered:** US by 10 (250 kHz; would cut the rpm-dependent content between 125 and 178 kHz).
**Rationale:** User's decision (2026-10-10), from the band check (D52). The user regards AE above 560 kHz as noise; US above 40 kHz may carry information (QUESTIONS.md, Q3).

## 2026-10-12 — D64: Downsampling on the fly; AE and US both at 1.25 MHz
**Decision:** (1) Downsampling is done when a run loads the data (not stored as a derived copy). (2) AE and US are both downsampled by 2, to 1.25 MHz, so the two channels stay simultaneous on one time grid and methods can use them together. Supersedes the US factor of D62 (by 5, to 500 kHz). Open: anti-alias filter; whether US gets an extra low-pass (e.g. at 178 kHz) to remove its mostly noisy content up to 625 kHz.
**Alternatives considered:** A stored downsampled copy (faster runs, but a second copy of the data on a nearly full disk). One model per sensor at its own rate (AE 1.25 MHz, US 500 kHz). Only one sensor per study.
**Rationale:** User's decision (2026-10-10). Simultaneous multichannel recordings are meant to be an advantage.
