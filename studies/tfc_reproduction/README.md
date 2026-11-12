# studies/tfc_reproduction — TF-C on FD-A → FD-B (D35–D46)

Status: closed as a record. Neither the official 2023 code nor our TF-C reproduces the published
0.8934 (official: loss NaN in 4 of 5 seeds on the full data, one class with the debugging subset;
ours: 0.706 ± 0.112), and FD-B is solved by the plain spectrum (0.999), so it cannot separate
methods (D45, D46, D70). Open: FD-A → FD-B transfer with the four TF-C variants (D211).

The FD-A/FD-B windows are used as distributed by the TF-C authors (dataset `fd`, D39). Scripts and
their configs were moved here from `scripts/` and `configs/` by D262; run them from the repository root.

| script | config (in `configs/` here) | what |
|---|---|---|
| `tfc_official.py` | `tfc_official_fd.yaml`, `tfc_official_fd_subset.yaml` | official TF-C code, full data / debugging subset; official and clean protocol (D36–D38, D43, D44) |
| `tfc_ours.py` | `tfc_fd.yaml` (paper loss), `tfc_positives_fd.yaml` (D122) | our TF-C, pretraining + fine-tuning (D41, D42) |
| `probe_tfc.py` | `probe_tfc_fd.yaml` | frozen probes on our TF-C encoders, with untrained reference (D45) |
| `probe_spectrum.py` | `probe_spectrum_fd.yaml` | probes on the plain \|FFT\| spectra of FD-B (D46) |

    uv run python studies/tfc_reproduction/tfc_official.py studies/tfc_reproduction/configs/tfc_official_fd.yaml

Method details: `docs/methods/tfc.md`; the FordA ablation is `studies/tfc_ablation/`.
