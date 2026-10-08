# Decisions — FordA (UCR archive)

Dataset-specific decisions for FordA. Index: `../../DECISIONS.md`.

## 2026-10-07 — D27: FordA used as distributed
**Decision:** FordA is used exactly as distributed in the UCR archive, with its official train/test split. The archive has already normalised every series to mean 0 and standard deviation 1; this transformation was applied before we received the data and is recorded here. No further normalisation is applied (TS2Vec's own code applies none to FordA either).
**Alternatives considered:** Re-normalising over the whole training set, as TS2Vec does for a list of non-normalised UCR datasets (FordA is not on that list).
**Rationale:** Matches the setting of the published results that the first slice is validated against.

## 2026-10-07 — D30: FordA metadata
**Decision:** FordA's class label (as in the file: −1 or 1) and its official split (`train`/`test`) are stored as extra columns on the captures table. The run's status is `final` (public benchmark). Source: the FordA archive from timeseriesclassification.com, plain-text files `FordA_TRAIN.txt` and `FordA_TEST.txt`.
**Alternatives considered:** A separate labels table (more structure than needed now).
**Rationale:** Per-capture facts belong with the capture; keeps stratification and splitting a filter on one table.
