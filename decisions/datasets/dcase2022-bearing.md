# Decisions — DCASE 2022 Task 2, machine type bearing

Dataset-specific decisions for the DCASE 2022 Challenge Task 2 (unsupervised anomalous sound detection for machine condition monitoring under domain shift), development data, machine type bearing (https://zenodo.org/record/6355122). Adopted by D189, downloaded by D194 (`data/raw/dcase2022_bearing/dev_bearing/bearing/`). Index: `../../DECISIONS.md`.

## 2026-11-10 — D251: DCASE 2022 bearing conversion and official split
**Decision:** Code: `harness/data/dcase2022_bearing.py`, `scripts/convert_dcase2022_bearing.py`; test `tests/test_dcase2022_bearing.py`.
1. *Hierarchy:* one run per section (`section_00` rotation-velocity shift, `section_01` microphone-location shift, `section_02` factory-noise shift; task page); each 10 s clip one capture (capture ID = file name without `.wav`); one stream `audio`, one channel `microphone`.
2. *Signals:* 16-bit PCM, mono, 16 kHz, 160,000 samples (all 3,599 clips checked, 2026-11-10). Stored as the int16 sample values exactly as in the WAV files, scale 1/32768, offset 0 (D6, D20), units "full scale (uncalibrated)". Read with Python's `wave` module (no new dependency).
3. *Per-capture columns (D31):* section, domain (source/target), split (train/test), label (normal/anomaly), clip index, the file name's attributes (JSON, e.g. `{"vel": "12", "loc": "A"}`), and the attribute CSV's `d1p` and `d1v` verbatim. The CSV and the file names agree for all 3,599 clips (checked); a disagreement stops the conversion. Clip counts per domain/split/label are recorded in the run extras (section 01 has 9 target training clips, D194).
4. *Split:* the official development split, as distributed: `train/` (normal clips only, 990 source + 10 target per section) for training and pretraining, `test/` (100 normal + 100 anomalous per section) for testing. Metrics as the task: AUC per section and domain, pAUC (p = 0.1) per section, and their harmonic mean. Running it needs the anomaly-detection scorers (D192). The additional training and evaluation datasets stay out (D189).
5. *Status and source:* `final`; D24 per run (name pattern, total size, latest modification time). Licence as stated on the Zenodo record, not yet checked.
**Alternatives considered:** One run per clip (3,599 runs). One run per section and split. Storing float values (the WAV holds integers).
**Rationale:** User's decision (2026-11-10): approved Claude's proposals as given ("go ahead").
