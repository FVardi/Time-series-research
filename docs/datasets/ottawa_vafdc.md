# University of Ottawa UODS-VAFDC

**Source:** Sehri, Dumond & Bouchard (2023, Data in Brief 49, 109327); Mendeley Data y2px5tg92h,
CC BY 4.0. 20 bearings, each healthy / fault developing / faulty; accelerometer and microphone at
42 kHz. Converted: one run per bearing, the three stages as captures, float64 as recorded
(D249). C_19_1 and C_19_2 are identical (recorded).

## Current split and task (D250: the dataset authors' protocol)
Sehri, Dumond et al. (2025, IJPHM, Table 3): ten domains of five files (one per class H, I, O, B,
C); domains 2, 4, 6, 8 for training and validation, domain 10 for test; five classes; no test
bearing in training.
- **Open:** training/validation division, windows and hop (paper: 1,024-sample inputs),
  normalisation (harness standard applies, P7), use of the odd (developing) domains for
  pretraining, vibration / acoustic / both.

History: D189, D194, D249, D250.
