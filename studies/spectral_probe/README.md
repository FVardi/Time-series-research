# Spectral probe (D244; checklist task A)

Do the SSL representations hold the spectrum of their input, in a form a linear probe can read?
Nothing is retrained: the stored representations (`seed_0/Z_*.npy`) of existing runs are probed for
spectral targets computed from the windows each encoder received (rebuilt with the run's own config
and checked against its `windows.csv`).

Targets per window and channel: log band energy (16 equal-width bands from 0 to Nyquist), log share of
the total energy per band, log total energy, spectral centroid. FordA and XJTU-SY use only the relative
energies and the centroid (their total energy is constant by construction). One ridge probe per target,
at 1, 10 and 100 % of the training windows, scored by test R²; floor and FFT reference rows.

One config per dataset in `configs/`, listing the runs (add runs there as they are made):

    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/forda.yaml
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/cwru.yaml
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/hust.yaml
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/paderborn.yaml
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/xjtu_sy.yaml

Writes `results/<config name>.csv`: one row per run (or `fft_reference`), label fraction, probe
(`ridge`, `floor_mean`), channel and target, with the band edges (cycles per sample, and Hz where
the sample rate is known), the chosen alpha and the test R². Figures: a notebook here (outputs cleared
before committing).
