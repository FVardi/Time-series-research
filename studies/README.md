# studies/ — one folder per study (D11, D91)

A study asks one question, has its own scripts and configs, imports only the harness (the harness
never imports from studies) and keeps figures in notebooks with outputs cleared (D100). Each
README starts with a `Status:` line, which `scripts/status.py` copies into `STATUS.md`.

| study | question |
|---|---|
| [spectral_probe](spectral_probe/README.md) | Do SSL representations keep their input's spectrum in a linearly readable form? (D244–D246, D252) |
| [tfc_ablation](tfc_ablation/README.md) | Which parts of TF-C's time–frequency consistency help, and where does it collapse? (D95–D97, D101) |
| [tfc_reproduction](tfc_reproduction/README.md) | Does TF-C reproduce on FD-A → FD-B? (D35–D46) |
| [split_pilot](split_pilot/README.md) | Can a representation block be freed of the operating point? (own method M1, parked) |
| [tsjepa_epochs](tsjepa_epochs/README.md) | Does TS-JEPA need its 5,000 epochs? (D200) |
| [method_checks](method_checks/README.md) | One-off checks of single methods (pooling, timing) |

The method benchmark itself is not a study folder: it is the standard runs (`configs/<method>_<dataset>.yaml`,
D239), compared with `scripts/compare.py` and `configs/compare/` (D203); its progress is in `STATUS.md`.
