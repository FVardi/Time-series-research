# DCASE 2022 Task 2, machine type "bearing" (development data)

**Source:** zenodo.org/record/6355122; sections 00 (rotation-velocity shift), 01 (microphone
location), 02 (factory noise); 10 s clips, 16 kHz, mono. Converted: one run per section, one capture
per clip, int16 PCM codes (D251). Section 01 has 9 target training clips (as distributed, D194).

## Current split and task (D251: official split)
- Anomaly detection trained on normal sounds: `train/` (990 source + 10 target normal clips per
  section) for training and pretraining, `test/` (100 normal + 100 anomalous per section).
- Scorers (D192): 1-NN and Mahalanobis on frozen embeddings, clip score = mean of window scores;
  metrics AUC per section and domain, pAUC (p = 0.1) per section, official harmonic mean.
- **Open:** the scorers are not implemented; windows not set.

History: D189, D192, D194, D251.
