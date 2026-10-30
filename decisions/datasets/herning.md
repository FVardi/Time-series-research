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

## 2026-10-09 — D51: Rig input representation: band-limit and downsample, cut-offs after checking where the information is (superseded by D165)
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

## 2026-10-12 — D64: Downsampling on the fly; AE and US both at 1.25 MHz (not used for the SSL input since D165)
**Decision:** (1) Downsampling is done when a run loads the data (not stored as a derived copy). (2) AE and US are both downsampled by 2, to 1.25 MHz, so the two channels stay simultaneous on one time grid and methods can use them together. Supersedes the US factor of D62 (by 5, to 500 kHz). Open: anti-alias filter; whether US gets an extra low-pass (e.g. at 178 kHz) to remove its mostly noisy content up to 625 kHz.
**Alternatives considered:** A stored downsampled copy (faster runs, but a second copy of the data on a nearly full disk). One model per sensor at its own rate (AE 1.25 MHz, US 500 kHz). Only one sensor per study.
**Rationale:** User's decision (2026-10-10). Simultaneous multichannel recordings are meant to be an advantage.

## 2026-10-30 — D165: Herning SSL input: a sequence of per-window band features (supersedes D51)
**Decision:** The SSL methods get, per sweep, a multichannel sequence of features computed on short windows, instead of the band-limited, downsampled signal (D51):
1. *Bands:* the feature-baseline octave bands, AE 50–100, 100–200, 200–400, 400–560 kHz (D61) and the eight UL base-10 octaves with centres 1.0 … 126 kHz, edges centre·10^(±0.15) (D60): 12 bands. All eight UL bands are kept, including the lowest three, whose Hjorth parameters and kurtosis are mostly noise at this window length (user: no preference; Claude's recommendation).
2. *Band signals:* order-5 zero-phase Butterworth band-pass per band on the whole sweep at the recorded 2.5 MHz (`bandpass_filter` of the MSSP code, `harness/methods/mssp_features.py`), then windows. The D64 downsampling is not used on this path (user: no preference; Claude's recommendation).
3. *Windows:* consecutive, non-overlapping windows of 1 ms (2,500 samples); the same windows for all features; a 0.2 s sweep gives 200 windows (incomplete last window dropped).
4. *Features per band and window (4):* log energy (natural log of the mean square of the band signal), Hjorth mobility and complexity, and kurtosis, the last three exactly as the MSSP code (antropy `hjorth_params`, mobility in samples⁻¹; Pearson kurtosis, 3 for Gaussian data). 12 bands × 4 = 48 channels; one sweep = one series of 200 × 48.
5. *Normalisation:* each of the 48 channels standardised with one mean and SD from the training windows (as D137); the log is the only other transform.
6. *Augmentations:* each SSL method's standard augmentations, unchanged.
Code: `harness/band_features.py`. Open: the Herning split (D56, QUESTIONS.md Q1) and wiring into the windowed pipeline (D147).
**Alternatives considered:** Band-limited, downsampled signal (D51/D64; too long for the methods). Log band energies only, on finer sub-bands (`eda/herning/envelope_model.ipynb` sections 10–11; hundreds of channels). Third-octave bands. 0.5 ms windows (400 steps, keeps more of the fast intensity variation, noisier kurtosis/Hjorth). Dropping the UL bands below 5.6 or 11.2 kHz. Filtering after the D64 downsampling; FFT band masks as in the notebook.
**Rationale:** User's decision (2026-10-30), after the envelope-model check (`eda/herning/envelope_model.ipynb`): within-sweep band energy dynamics vary with speed and track κ within speed; the scale scan put most of that variation at 0.1–5 ms, of which 1 ms windows keep about 75–85 %. The number of independent values per window is about 2·B·T whatever the sampling rate, so not downsampling does not make the features steadier; 1 ms gives about 22 for UL 11–22 kHz and 100–400 for the AE bands.

## 2026-10-30 — D169: Herning validation split: random 80/20 of the sweeps with κ, one fixed draw
**Decision:** For the validation of the Herning path, the sweeps with a κ label (D55: excluded sweeps have none) are split at random, 80 % training and 20 % test, in one fixed draw (seed 0); each sweep is one series (D165). Pretraining uses the training sweeps only (as D164 for CWRU); the bearing-still sweeps are not used. Config: `split: {captures: {}, random: {test_fraction: 0.2, seed: 0}}` (`harness/windows.py`, `random_parts`); first config `configs/pooled_herning.yaml`. Settles D56's open details for this validation only; the temperature hold-out (D56) remains the intended test.
**Alternatives considered:** The temperature hold-out of D56 (block(s) still to choose). Bearing-still sweeps as unlabelled pretraining data. 70/30 or 50/50. A new draw per seed.
**Rationale:** User's decision (2026-10-30): a validation run, to check that the path works. Caveat: neighbouring sweeps of one hold are near-duplicates and land on both sides, so test scores are optimistic.

## 2026-10-30 — D172: Herning band features stored once, read by every run (script generalised and renamed by D181)
**Decision:** The per-window band features (D165) of every Herning sweep (labelled or not) are computed once by `scripts/derive_band_features.py` and stored in `data/derived/herning/band_features.h5` (float64, the values the pipeline would compute; about 300 MB; not in git, as all of `data/`). Configs read them with `windows: features_file` (`harness/windows.py`, `cut_stored`). The file records the settings that determine the features (channels, sample rate, window, features, bands, filter); reading stops if they differ from the code's, so a changed feature definition needs a rerun of the script. Checked: the stored path gives exactly the same arrays and window tables as computing on the fly (120 converted sweeps, 2026-10-30). The on-the-fly path stays for configs without `features_file`.
**Alternatives considered:** Computing the features in every run (about 30 minutes per run on the full data). float32 (half the size; values then differ slightly from the on-the-fly path).
**Rationale:** User's decision (2026-10-30): the file is small. Unlike the stored downsampled signal that D64 ruled out for disk space, the features are about 0.1 % of the raw data.

## 2026-10-30 — D173: SSL methods on Herning: official settings, changed only where the input shape forces it (as D151 on CWRU)
**Decision:** As on CWRU (D151): every method runs with its standard settings, changed only where the input (one series of 200 steps × 48 channels per sweep, D165) forces it; data-dependent official rules applied; nothing searched; a method that clearly fails to train is looked at case by case and recorded. The multichannel adaptations decided for CWRU apply unchanged (D154 TS-JEPA, D155 TF-JEPA, D156 TS-TCC permutation, D159 SimMTM). Shape-forced values (`configs/<method>_herning.yaml`):
1. *TS-TCC:* `input_channels` 48, `features_len` 27 (encoder output length for 200 steps; checked by the adapter), `num_classes` 1 (sizes the unused classifier layer only; κ is regressed by the probe).
2. *SimMTM:* `TSlength_aligned` 200, `CNNoutput_channel` 10 (checked by the adapter), `input_channels` 48; the decoder rebuilds 48 × 200 values (D159).
3. *TS-JEPA:* 10 patches (paper) of 20 steps per channel, none unused; 20 is not a multiple of the tokenizer's stride 3, so some tokenizer windows span two channels within a flattened patch (CWRU's 204 was a multiple).
4. *TF-JEPA:* model width = series length 200 (feed-forward 400); each of the 48 channels through the same model; representation 48 × 256 values (D155).
5. *PatchTST:* patch length 12 unchanged: 16 patches, the first 8 steps of each series unused. Note: RevIN normalises each channel of each series by its own mean and SD, so PatchTST does not see the per-sweep level of the features (e.g. the band energy level), only their variation within the sweep.
6. *TS2Vec:* official rule for the number of iterations gives 600 (about 2,960 × 200 × 48 values > 100,000); `max_train_length` 3,000 > 200, no cropping. *T-Loss:* unchanged. *FreRA, TF-C (positives):* unchanged, sized from the data (TF-C: 128 × 5 per branch, h = 640).
**Alternatives considered:** Choosing settings per method for the new input (as D88 left open for Herning).
**Rationale:** User's decision (2026-10-30): the same philosophy as CWRU. Values in items 1–6 follow from the input shape by the rules used on CWRU (Claude, from the adapters' formulas); items 3 and 5 are side effects of applying them unchanged, recorded here.
