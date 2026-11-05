# Decisions — XJTU-SY bearing data

Dataset-specific decisions for the XJTU-SY run-to-failure bearing data (Xi'an Jiaotong University and Changxing Sumyoung Technology; https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets). Index: `../../DECISIONS.md`. Adopted by D189.

## 2026-11-02 — D191: XJTU-SY tasks and derived labels
**Decision:** XJTU-SY is used for several tasks on the same frozen representations, each with the harness probe that fits it. The dataset provides run-to-failure recordings, not health labels; all labels below are derived in the pipeline from the recording times, and the stored data stay as recorded (D5).
1. *Fault element* (classification): the failed element per bearing, if the dataset's documentation lists it (to be confirmed at download).
2. *Remaining useful life* (regression, D58/D59 probe): piecewise-linear target — constant before the first prediction time (FPT), then decreasing linearly to 0 at the end of the run; normalised to [0, 1] per bearing. Metrics: the regression metrics of the harness (MAE, R²) plus RMSE.
3. *Health stage / detection* (classification, and the anomaly protocol of D192): two stages, healthy (before FPT) and degrading (after FPT).
4. *Health indicator* (label-free): the D192 anomaly score of each window against the bearing's own early-life windows, evaluated with monotonicity, trendability and prognosability (Coble & Hines, 2009; exact definitions to be taken from the source at implementation).
FPT (one definition for tasks 2–4): the 3σ rule on the root-mean-square value — FPT is the first time the RMS exceeds the mean + 3 standard deviations of an early-life reference segment. Open, settled after looking at the data: the reference segment length, the channel(s) used, and how many consecutive exceedances are required (guard against single spikes).
Split by bearing (D121); conditions and number of held-out bearings decided with the configuration.
**Alternatives considered:** Linear remaining-useful-life target from the start of the run (penalises healthy early life). More than two health stages (no common definition). FPT by fixed fractions of life (ignores the large spread of bearing lives).
**Rationale:** User delegated the choice to Claude (2026-11-02), asking for the most widely used and/or most flexible options. The piecewise-linear target with an FPT from a 3σ rule is a common choice in bearing prognostics; one FPT definition keeps the tasks consistent; all labels are derived, so they can be changed without reconverting.

*Note (2026-11-03): task 1 confirmed. Table 2 of `Introduction_to_XJTU-SY_Bearing_Dataset.pdf` (in the archive) lists the fault element per bearing: outer race 8 (1_1, 1_2, 1_3, 2_2, 2_4, 2_5, 3_1, 3_5), inner race 3 (2_1, 3_3, 3_4), cage 2 (1_4, 2_3), combined 2 (1_5: inner and outer race; 3_2: inner race, ball, cage and outer race). Classes are small and unbalanced for a split by bearing, and the label describes the end of life, not the healthy start; both to be handled in the task configuration. Bearing type LDK UER204; test ended when the vibration amplitude exceeded 10 × the normal-stage maximum.*

## 2026-11-03 — D193: XJTU-SY raw data stored as downloaded (packing superseded by D213)
**Decision:** The XJTU-SY data are stored unchanged in `data/raw/xjtu_sy/` as downloaded from the authors' Google Drive (linked from https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets): the six-part RAR archive (`Data/`, 9,236 CSV files, 12.2 GB unpacked; one CSV per minute, horizontal and vertical vibration, 32,768 samples at 25.6 kHz), the read-me files, `Important References/` and `Photographs/`. The archive is kept packed, as for Paderborn (D47); the conversion reads from it. No licence is stated; publications must cite Wang, Lei, Li and Li (2020), IEEE Transactions on Reliability 69(1), 401–412, doi 10.1109/TR.2018.2882682.
**Alternatives considered:** Unpacking the archive (12.2 GB of CSV next to the 4.4 GB archive).
**Rationale:** Follows D47 and D5 (raw data as recorded). Downloaded by the user (2026-11-03).

*Note (2026-11-04): the data are now unpacked in `data/raw/xjtu_sy/XJTU-SY_Bearing_Datasets/<condition>/<bearing>/<n>.csv` (D213); the archive parts were deleted.*
