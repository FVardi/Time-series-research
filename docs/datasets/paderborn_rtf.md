# Paderborn run-to-failure data under time-varying conditions (LDM)

**Source:** Aimiyekagbon (2024), Zenodo 10868257, CC BY 4.0; cite Javanmardi et al. (2024).
17 experiments (B01–B17), new 61806-2RS bearings run to failure, two accelerometers (A, C), load and
speed varied at random per measurement (D234). Not the KAt fault data.
Converted: one run per experiment, one capture per 1.6 s measurement, channels as int16 ADC codes
(bit-exact checks per file), telemetry from the operating-conditions CSVs, units V (D236). Raw
vibration files are not kept; Zenodo is the archive (D235). 36 GB canonical.

## Current split and task
- Standard by D214/D216: the authors' protocol (Javanmardi et al., 2024): leave one bearing out over
  all 17 experiments, remaining useful life after a transition time.
- **Open:** task, labels (transition time, warm-up phase) and split details.

History: D234–D236.
