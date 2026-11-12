# Spectral probe (D244; checklist task A)

Status: active (checklist task A). Probes and premise tests (a) and (c) run on FordA, CWRU, HUST and Paderborn; XJTU-SY rerun, premise test (b) and the rows of new runs are open.

Do the SSL representations hold the spectrum of their input, in a form a linear probe can read?
Nothing is retrained: the stored representations (`seed_0/Z_*.npy`) of existing runs are probed for
spectral targets computed from the windows each encoder received (rebuilt with the run's own config
and checked against its `windows.csv`).

Targets per window and channel: log band energy (16 equal-width bands from 0 to Nyquist), log share of
the total energy per band, log total energy, spectral centroid. FordA and XJTU-SY use only the relative
energies and the centroid (their total energy is constant by construction). One ridge probe per target,
trained on all training windows (alpha from 10⁻⁶…10⁸, D246), scored by test R²; floor and FFT reference rows.

One config per dataset in `configs/`, listing the runs (add runs there as they are made):

    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/forda.yaml
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/cwru.yaml
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/hust.yaml
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/paderborn.yaml
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/xjtu_sy.yaml

Premise tests (a) (D245) and (c) (D252): the spectral targets themselves as features for the benchmark's
logreg probe, all together (a) and each band's relative energy on its own (c), with the same configs:

    uv run --no-sync python studies/spectral_probe/classify_spectral.py studies/spectral_probe/configs/forda.yaml

(and `cwru`, `hust`, `paderborn`, `xjtu_sy`), written to `results/<config name>_classify.csv` (a) and `results/<config name>_bands.csv` (c).
Premise test (b) and the figures: `spectral_probe.ipynb` (run from the repo root; clear the outputs before committing).

`probe_spectral.py` writes `results/<config name>.csv`: one row per run (or `fft_reference`), label fraction, probe
(`ridge`, `floor_mean`), channel and target, with the band edges (cycles per sample, and Hz where
the sample rate is known), the chosen alpha and the test R². With `per_bearing: true` (XJTU-SY, D265) also `results/<config name>_per_bearing.csv`:
Pearson's r between predictions and targets within each test bearing.
