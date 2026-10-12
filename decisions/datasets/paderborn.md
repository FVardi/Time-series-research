# Decisions — Paderborn bearing data (raw)

Dataset-specific decisions for the raw Paderborn bearing dataset (KAt DataCenter, Lessmeier et al., 2016). Index: `../../DECISIONS.md`.

## 2026-10-08 — D47: Download all 32 Paderborn bearing archives
**Decision:** All 32 archives (6 healthy, 12 artificially damaged, 14 really damaged bearings; one `.rar` per bearing, about 160 MB each, about 5.1 GB in total) plus `readme_versions.txt` are downloaded unchanged from https://groups.uni-paderborn.de/kat/BearingDataCenter/ into `data/raw/paderborn/`. Licence CC BY-NC 4.0; the source must be cited (Lessmeier et al., 2016, PHM Europe; KAt-DataCenter, Paderborn University). Conversion is decided separately.
**Alternatives considered:** Only the bearings needed for a first experiment (splits by bearing need many bearings; the full set is small).
**Rationale:** User's decision (2026-10-08). Step 2 of D35 (TF-C on the full Paderborn data, splits by bearing).

## 2026-10-08 — D48: Paderborn signals to convert: vibration and both motor currents
**Decision:** When the data are converted, the housing vibration and the two motor phase currents (all 64 kHz) are converted. Speed, torque, radial force and temperature (lower rates) are not converted for now. The conversion itself (mapping onto D1, metadata, timing) is decided separately.
**Alternatives considered:** Vibration only (the signal FD-A/FD-B use, according to the TF-C paper). All signals as recorded.
**Rationale:** User's decision (2026-10-08).
