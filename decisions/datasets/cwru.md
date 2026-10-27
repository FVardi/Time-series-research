# Decisions — CWRU bearing data

Dataset-specific decisions for the CWRU bearing data (Case Western Reserve University Bearing Data Center, https://engineering.case.edu/bearingdatacenter). Index: `../../DECISIONS.md`.

## 2026-10-27 — D116: Convert all CWRU files; subsets chosen per experiment
**Decision:** All data files published by the Bearing Data Center are converted to the canonical format (D3, D5): normal baseline, 12 kHz drive-end and fan-end faults, 48 kHz drive-end faults, all fault sizes and outer-race positions. Which files an experiment uses is set in its config (D16). Download, mapping onto D1 and metadata are decided separately.
**Alternatives considered:** Converting only the usual 10-class subset (drive end, 12 kHz; inner race, ball and outer race @6:00 at 7, 14 and 21 mil; normal).
**Rationale:** User's decision (2026-10-27). Any subset can be chosen later without a second conversion.

## 2026-10-27 — D117: CWRU 28 mil faults not used in experiments
**Decision:** The 28 mil fault files are converted (D116) but not used in experiments.
**Alternatives considered:** Using them (one more inner-race and one more ball bearing at the drive end for splits by bearing).
**Rationale:** User's decision (2026-10-27). According to the data center's fault specifications, the 7, 14 and 21 mil faults are in SKF bearings with fault depth 0.011 in, the 28 mil faults in NTN bearings with depth 0.050 in (inner race) and 0.150 in (ball), so fault size would be confounded with manufacturer and depth.

## 2026-10-27 — D118: CWRU outer-race positions kept apart (storage of the position settled by D120)
**Decision:** Outer-race faults at the three positions given by the data center (@3:00, @6:00, @12:00) are kept as separate fault conditions; they are not merged into one outer-race class. How the position is stored is decided with the metadata.
**Alternatives considered:** Merging them into one outer-race class (more outer-race bearings per class). Using @6:00 only (the usual subset).
**Rationale:** User's decision (2026-10-27). Outer-race faults are stationary, so their position relative to the load zone changes the vibration response (data center, Apparatus & Procedures).
*Note:* the data center's documentation is inconsistent about which position is in the load zone: the data-file tables label @6:00 "Centered", @3:00 "Orthogonal" and @12:00 "Opposite", while the Apparatus & Procedures page describes the 6 o'clock position as "orthogonal to the load zone".

## 2026-10-27 — D119: CWRU 40 mil faults not used in experiments
**Decision:** As D117 for 28 mil: any 40 mil fault files are converted (D116) but not used in experiments. The fault specifications list a 40 mil drive-end outer-race fault; it does not appear in the 12 kHz drive-end file table, so whether files exist is checked at download.
**Alternatives considered:** Using them.
**Rationale:** User's decision (2026-10-27). Same confound as D117: NTN bearing, fault depth 0.050 in (SKF, 0.011 in for 7–21 mil).

## 2026-10-27 — D120: CWRU outer-race position stored as the recorded clock hour
**Decision:** The outer-race fault position (D118) is stored in the metadata as the clock hour given in the data center's file tables (3, 6 or 12). The descriptions "Centered", "Orthogonal" and "Opposite" are not stored, because the documentation is inconsistent about which position is in the load zone (note under D118).
**Alternatives considered:** Storing the descriptions from the file tables.
**Rationale:** User's decision (2026-10-27). Values as recorded (D5); the interpretation stays out of the stored data.

## 2026-10-27 — D125: No healthy class in CWRU experiments
**Decision:** CWRU experiments classify faults only; the healthy class (normal files 97–100) is not used as a class. The normal files are still converted (D116). Detection (healthy vs faulty) is tested on Paderborn, which has six healthy bearings. Consequence for D121: the test bearings are the held-out faulty bearings, and the files of a test bearing (one per load) are left out of pretraining; the two healthy bearings are in every fault file, on both sides of any split, as background, not as a class. The label set (fault type, outer-race position, size) is decided separately.
**Alternatives considered:** Keeping the healthy class: CWRU has one healthy bearing per end, and every fault file also measures the healthy bearing at the other end, so the healthy class cannot be split without leakage. Splitting the files into one series per channel does not help: in the cross-talk check (`eda/cwru/check_crosstalk.py`, `runs/20261027_131116_crosstalk_cwru`) the opposite channel shows the fault's defect frequency at a median 27.5 dB above the normal reference and a median 2.8 dB below the fault's own channel (59 files where the fault is clearly visible in its own channel).
**Rationale:** User's decision (2026-10-27). CWRU's role is fault diagnosis as a public benchmark of representation quality; detection is covered cleanly by Paderborn.

## 2026-10-27 — D126: CWRU normal files used for pretraining (rate matching settled by D136)
**Decision:** The normal files (97–100) may be used in self-supervised pretraining (and other fitting without labels) on CWRU. They contain only the two healthy bearings, which are not test bearings (D125), so D121 allows it. The normal files are sampled at 48 kHz (`eda/cwru/check_normal_fs.py`, `runs/20261027_131034_normal_fs_cwru`), the fault files at 12 kHz; how the rates are matched for pretraining is decided with the pretraining setup.
**Alternatives considered:** Pretraining on fault files only.
**Rationale:** User's decision (2026-10-27).

## 2026-10-27 — D127: CWRU split: one fault size per class and end held out, repeated random draws (superseded by D128)
**Decision:** CWRU is one classification task over both ends (drive end and fan end pooled), with five fault classes: inner race, ball, outer race @3, @6, @12 (D118; no healthy class, D125; 7, 14 and 21 mil only, D117, D119). The split is by bearing: for each class and end, one fault size is drawn at random and its bearing (all its files, every load) goes to test; the other bearings of that class and end go to training. Test bearings' files are left out of pretraining (D121); the normal files may be used (D126). The split is repeated over several random draws, each with its own pretraining. Open: the number of draws; class–end cells with a single bearing (fan-end outer race @12: 7 mil only); how probe cross-validation inside the training set is grouped.
**Alternatives considered:** One end only (two or three bearings per class; fan-end outer race @12 cannot be split). Holding out a whole end (transfer to the other bearing model; one harder split). Leave one bearing out (best use of the data, one pretraining per bearing).
**Rationale:** User's decision (2026-10-27). Pooling both ends gives three to six bearings per class; the split follows Rosa et al. (2024).

## 2026-10-27 — D128: CWRU split: train on 7 and 14 mil, test on 21 mil (supersedes D127) (sampling rate clarified by D140)
**Decision:** One fixed split by bearing. The task is unchanged from D127: both ends pooled, five fault classes (inner race, ball, outer race @3, @6, @12). Training: the 7 and 14 mil bearings of both ends (17 bearings). Test: the 21 mil bearings (9 bearings: drive end all five classes; fan end inner race, ball, outer race @6 and @3). The fan-end outer race @12 bearing (7 mil only) is in training; outer race @12 is tested on the drive end only. Test bearings' files are left out of pretraining (D121); the normal files may be used (D126). No random draws: runs differ by seed only, as on the other datasets. This is one particular draw of D127, so D127's open points on the number of draws and the single fan-end outer race @12 bearing are settled. Still open: how probe cross-validation inside the training set is grouped.
**Alternatives considered:** D127's repeated random draws (one pretraining per draw; more decisions). Split by end, e.g. train drive end, test fan end: also changes bearing type (other defect frequencies), sensor position and the available classes at once, so a failure cannot be attributed; possibly a second, harder test later.
**Rationale:** User's decision (2026-10-27). Tests generalisation to unseen bearings with bearing types, sensors and classes unchanged; 21 mil is the only size present in every drive-end class. It tests on the largest faults, i.e. generalisation to more severe faults, not to any unseen bearing.

*Note (2026-10-27): D125–D128 were first recorded as D122–D125. A concurrent edit used D122–D124 for other decisions (decisions/training.md), so these entries were renumbered; content unchanged apart from the decision numbers.*

## 2026-10-27 — D131: CWRU input: drive-end and fan-end channels as one two-channel series
**Decision:** On CWRU, every series given to a method has two channels, drive end (DE) and fan end (FE), recorded at the same time. The base-plate channel (BA) is not used (missing in the normal files, which are pretraining data, D126, and in some fault runs; still converted, D116). Open: MiniRocket, whose wrapper uses the official univariate code (wrap the official multivariate version, or report MiniRocket as not run on CWRU, D70).
**Alternatives considered:** Each channel as its own univariate series with the file's label (twice as many series, runs on every method; the sensor of a series is unknown to the model; the DE and FE windows of one moment are near-duplicates). Only the channel at the faulty bearing's end (choosing the input needs the fault location, which is part of the answer).
**Rationale:** User's decision (2026-10-27). Keeps both sensors' views together, including which sensor sees the fault more strongly; both carry the fault (cross-talk check, D125). Multichannel input is needed on Paderborn anyway (D48).

## 2026-10-27 — D136: CWRU windows: 2,048 samples, no overlap; normal files resampled to 12 kHz (open points settled by D137)
**Decision:** Methods and probes see windows of 2,048 samples (about 171 ms at 12 kHz, about five shaft revolutions; spectral resolution 5.9 Hz), without overlap, for pretraining, training and test windows alike. The normal files (48 kHz, D126) are resampled to 12 kHz before windowing with `scipy.signal.resample_poly` (factor 1/4, its built-in anti-alias filter), so that all windows have the same duration. Resampling is a transform in the pipeline; the stored data stay as recorded (D5). Open: where windows start in a file and what happens to the remainder; normalisation.
**Alternatives considered:** 1,024 samples (about 2.5 revolutions, 11.7 Hz resolution: too coarse to separate drive-end inner race and ball fault frequencies, about 157 and 137 Hz). 4,096 samples (half as many windows; costlier for some methods). 50 % overlap for pretraining and training windows (twice as many, strongly correlated windows).
**Rationale:** User's decision (2026-10-27). Several fault impacts and cage cycles per window; 2,048 is common in CWRU papers.

## 2026-10-27 — D137: CWRU window placement and per-channel standardisation with training statistics
**Decision:**
1. *Window placement:* windows are cut consecutively from the first sample of each file (after resampling for the normal files, D136); the incomplete last window of a file is dropped (fewer than 2,048 samples, under 2 % of a 10 s file).
2. *Normalisation:* each channel is standardised with one mean and one standard deviation, computed over all samples of the training and pretraining windows of the split (D128: the 7 and 14 mil fault files and the resampled normal files; never the test bearings) and applied unchanged to every window, training and test. This is a fixed change of units per channel: amplitude differences between windows, and the relation between the two channels within a window, are kept; only the overall ratio of the two sensors' raw scales changes, by one constant factor. On the D128 training data: drive end mean 0.0053 g, std 0.2812 g; fan end mean 0.0231 g, std 0.2433 g (computed with the cross-talk check's file reading; the implementation recomputes them). A transform in the pipeline; stored data stay as recorded (D5). The baselines keep their inputs as in D76 (raw values as loaded).
**Alternatives considered:** No normalisation (values in g, RMS 0.06–0.94 g, median about 0.15 g: the methods' official augmentation magnitudes, initialisations and learning rates assume inputs of roughly unit scale, so they would act differently than on the data they were tuned on; the louder channel would dominate reconstruction losses). Z-normalising each window on its own (removes amplitude, a main fault cue).
**Rationale:** User's decision (2026-10-27).

## 2026-10-27 — D138: CWRU label budgets drawn as on the other datasets
**Decision:** The label budgets 1 %, 10 % and 100 % (D40) are drawn as in D29: a random subset of the labelled training windows, stratified by class, new for each seed. On the D128 split there are about 3,780 labelled training windows (64 files), so about 38 labelled windows at 1 % and 378 at 10 %; at 1 % they spread over almost all 17 training bearings. The grouped cross-validation of D129 is built on the labelled windows. Known limitation, to be stated in the report: in practice labels usually come per bearing or recording, so a small budget would mean few bearings with many windows each; drawing windows at random spreads the labels over many bearings and is optimistic about label diversity.
**Alternatives considered:** Budgets counted in bearings (17 training bearings: 10 % is under two bearings, which cannot cover five classes). Stratified by class and spread evenly over the bearings of each class (less variance from the draw, even more optimistic about label diversity).
**Rationale:** User's decision (2026-10-27). Consistent with the other datasets.

## 2026-10-27 — D139: CWRU conversion
**Decision:**
1. *Hierarchy (D1, D3):* one run per source `.mat` file (one test: one bearing pair at one load), one capture per run, one stream per run (all channels share a clock), channels DE, FE and BA as present in the file.
2. *Per-capture columns (D31):* source file number, load (hp), fault end (DE/FE/none), fault location (inner/ball/outer/none), fault size (mil), outer-race clock hour (D120), fault depth (in), bearing manufacturer, and a bearing ID for each end (`de_bearing`, `fe_bearing`, e.g. `DE_IR_007`, `DE_healthy`). The bearing IDs are what D121 and D129 use. The split is not stored (set in the experiment config, D128).
3. *Speed:* the RPM value stored in the file where present; missing where the file has none (98.mat, the 28 mil files), not filled in. The data center's approximate table speed in a separate column.
4. *Sampling rate:* 12 or 48 kHz per the data center's file tables; the normal files 48 kHz, from `eda/cwru/check_normal_fs.py` (the data center does not state it), with that source noted in the metadata.
5. *Units:* `unknown`, with a note that the literature treats the values as g (not stated in the files or on the data center's pages read).
6. *Quirks:* 99.mat also holds file 98's variables (`X098_*`, byte-identical to 98.mat): only `X099_*` are converted. Stray variables (`ans` in 99.mat; `i`, a 10,000-element ±1 vector, in 277 and 278) are ignored and listed in the conversion log. The 28 mil files each hold one drive-end variable whose number does not match the file (3001–3004: X056–X059; 3005–3008: X048–X051): that variable is taken and its original name recorded.
7. *Status and type:* status `final` (public benchmark, as FordA, D30); values as 64-bit floats as in the files (D6).
8. *Source files:* downloaded unchanged into `data/raw/cwru/` from the data center (`https://engineering.case.edu/sites/default/files/<number>.mat`). Still to fetch: 145.mat (truncated) and the 48 kHz drive-end fault files.
**Alternatives considered:** One run for the whole dataset with one capture per file (one HDF5 file; "run" would no longer mean one test). Filling missing speeds with the table speed (not as recorded). Units stored as `g` (assumed, not documented).
**Rationale:** User's decision (2026-10-27).

## 2026-10-27 — D140: CWRU 48 kHz file quirks; experiments use the 12 kHz fault files
**Decision:**
1. *Clarification of D128:* CWRU experiments use the 12 kHz fault files (drive end and fan end) and, for pretraining, the normal files (D126, resampled to 12 kHz, D136). The 48 kHz drive-end fault files are converted (D116) but not used.
2. *Quirks of the 48 kHz files* (found by the conversion, which stops on unexpected variables), handled as D139 item 6:
   - 217.mat also holds file 215's variables (`X215_*`, byte-identical to 215.mat): only `X217_*` is converted.
   - 175.mat holds its own `X175_*` (381,890 samples, 8 s, shorter than its neighbours) plus `X217` (two columns) and `X217_DE_time`: rescaled copies of file 217's channels (correlation 1.000 with 217's DE and FE channels, about 0.2 times their amplitude; `X217_DE_time` matches 217's FE channel). Only `X175_*` is converted; the copies are noted in the run's extras.
   - 174.mat holds no `X174_*` but `X173_*` (DE, FE, RPM 1796) of only 63,788 samples (1.3 s, versus about 10 s for its neighbours). That variable set is converted, its original name recorded, and the short length and the possibility that it is not the intended recording noted in the run's extras.
**Alternatives considered:** Not converting 174.mat. Using the 48 kHz drive-end files in experiments (no 48 kHz fan-end data; windows of D136 are set for 12 kHz).
**Rationale:** User's decision (2026-10-27).

## 2026-10-27 — D141: CWRU speed where none is recorded: the table speed, marked as estimated
**Decision:** Wherever a CWRU shaft speed is needed (e.g. defect frequencies), the speed recorded in the file is used, otherwise the data center's approximate table speed for the load (`rpm_table`, D139 item 3), and the value is marked as estimated. A pipeline step; the stored data stay as recorded (D5). Affects runs 0098, 0099 and the 28 mil runs (3001–3008).
**Alternatives considered:** Leaving such runs out wherever speed is needed.
**Rationale:** User's decision (2026-10-27). In the 151 runs with a recorded speed, recorded minus table speed has median −1 rpm, 130 runs within ±5 rpm, up to 12 rpm slower at 3 hp (about 0.7 %); one outlier, run 0279 (fan-end inner race 7 mil, 1 hp): 1732 against 1772 rpm.
