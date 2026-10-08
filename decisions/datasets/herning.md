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
