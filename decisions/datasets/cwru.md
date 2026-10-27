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

## 2026-10-27 — D126: CWRU normal files used for pretraining
**Decision:** The normal files (97–100) may be used in self-supervised pretraining (and other fitting without labels) on CWRU. They contain only the two healthy bearings, which are not test bearings (D125), so D121 allows it. The normal files are sampled at 48 kHz (`eda/cwru/check_normal_fs.py`, `runs/20261027_131034_normal_fs_cwru`), the fault files at 12 kHz; how the rates are matched for pretraining is decided with the pretraining setup.
**Alternatives considered:** Pretraining on fault files only.
**Rationale:** User's decision (2026-10-27).

## 2026-10-27 — D127: CWRU split: one fault size per class and end held out, repeated random draws (superseded by D128)
**Decision:** CWRU is one classification task over both ends (drive end and fan end pooled), with five fault classes: inner race, ball, outer race @3, @6, @12 (D118; no healthy class, D125; 7, 14 and 21 mil only, D117, D119). The split is by bearing: for each class and end, one fault size is drawn at random and its bearing (all its files, every load) goes to test; the other bearings of that class and end go to training. Test bearings' files are left out of pretraining (D121); the normal files may be used (D126). The split is repeated over several random draws, each with its own pretraining. Open: the number of draws; class–end cells with a single bearing (fan-end outer race @12: 7 mil only); how probe cross-validation inside the training set is grouped.
**Alternatives considered:** One end only (two or three bearings per class; fan-end outer race @12 cannot be split). Holding out a whole end (transfer to the other bearing model; one harder split). Leave one bearing out (best use of the data, one pretraining per bearing).
**Rationale:** User's decision (2026-10-27). Pooling both ends gives three to six bearings per class; the split follows Rosa et al. (2024).

## 2026-10-27 — D128: CWRU split: train on 7 and 14 mil, test on 21 mil (supersedes D127)
**Decision:** One fixed split by bearing. The task is unchanged from D127: both ends pooled, five fault classes (inner race, ball, outer race @3, @6, @12). Training: the 7 and 14 mil bearings of both ends (17 bearings). Test: the 21 mil bearings (9 bearings: drive end all five classes; fan end inner race, ball, outer race @6 and @3). The fan-end outer race @12 bearing (7 mil only) is in training; outer race @12 is tested on the drive end only. Test bearings' files are left out of pretraining (D121); the normal files may be used (D126). No random draws: runs differ by seed only, as on the other datasets. This is one particular draw of D127, so D127's open points on the number of draws and the single fan-end outer race @12 bearing are settled. Still open: how probe cross-validation inside the training set is grouped.
**Alternatives considered:** D127's repeated random draws (one pretraining per draw; more decisions). Split by end, e.g. train drive end, test fan end: also changes bearing type (other defect frequencies), sensor position and the available classes at once, so a failure cannot be attributed; possibly a second, harder test later.
**Rationale:** User's decision (2026-10-27). Tests generalisation to unseen bearings with bearing types, sensors and classes unchanged; 21 mil is the only size present in every drive-end class. It tests on the largest faults, i.e. generalisation to more severe faults, not to any unseen bearing.

*Note (2026-10-27): D125–D128 were first recorded as D122–D125. A concurrent edit used D122–D124 for other decisions (decisions/training.md), so these entries were renumbered; content unchanged apart from the decision numbers.*

## 2026-10-27 — D131: CWRU input: drive-end and fan-end channels as one two-channel series
**Decision:** On CWRU, every series given to a method has two channels, drive end (DE) and fan end (FE), recorded at the same time. The base-plate channel (BA) is not used (missing in the normal files, which are pretraining data, D126, and in some fault runs; still converted, D116). Open: MiniRocket, whose wrapper uses the official univariate code (wrap the official multivariate version, or report MiniRocket as not run on CWRU, D70).
**Alternatives considered:** Each channel as its own univariate series with the file's label (twice as many series, runs on every method; the sensor of a series is unknown to the model; the DE and FE windows of one moment are near-duplicates). Only the channel at the faulty bearing's end (choosing the input needs the fault location, which is part of the answer).
**Rationale:** User's decision (2026-10-27). Keeps both sensors' views together, including which sensor sees the fault more strongly; both carry the fault (cross-talk check, D125). Multichannel input is needed on Paderborn anyway (D48).
