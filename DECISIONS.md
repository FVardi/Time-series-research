# Decision log (from D270)

Why the harness is the way it is. What holds now is in `docs/` (each rule cites its D-number).
D1–D269 are in `decisions_archive/` (frozen 2026-11-17).

## What gets an entry
A choice the user made, or a departure from the standard, that affects results or what a
study can claim. Not here:
- choices taken from a standard: config comment and the method's or dataset's card in `docs/`;
- working notes on runs: the study's or run's README;
- how Claude works: `CLAUDE.md`.

Entries are not edited afterwards, except their status: a replaced decision is marked
"Superseded by Dn" in the entry and in the index.

## Template (about 10 lines at most)

    ## Dn — Short title (YYYY-MM-DD)
    **Status:** Accepted | Superseded by Dm
    **Context:** one or two sentences: what question or problem.
    **Decision:** what was decided.
    **Alternatives:** other options and why not.
    **Consequences:** what changes (cards, configs, runs).

## Index

| ID | Date | Decision | Status |
|---|---|---|---|
| D270 | 2026-11-18 | Spectral probe: class contrast per band, η² of the relative band energy over the training windows (signal and labels only); each run's band R² against η², Spearman's ρ (`band_eta2.py`), instead of hand-picked bands | Accepted |
| D271 | 2026-11-18 | Spectral probe notebook only on band R² (SSL representations vs FFT) and η²; premise tests (a), (c) no longer shown, (b) removed; new `band_spectra.ipynb` (class spectra with the bands overlaid) | Accepted |

## Entries

## D270 — Spectral probe: which bands carry class information (2026-11-18)
**Status:** Accepted
**Context:** The user wanted the probes to predict specific bands rather than all bands alike, without
using any probe accuracy on the representations, and with one rule for all datasets.
**Decision:** Rank the 16 bands of D244 by η² (between-class / total sum of squares) of their log relative
energy over each run's training windows (Paderborn: per combination); per run, Spearman's ρ between band
R² and band η² over all bands and channels, read against the FFT reference's ρ. Existing probe results
are used; nothing is retrained or re-probed.
**Alternatives:** hand-picked bands per dataset (physics or inspection; not one rule for all datasets);
finer grid, fixed top/bottom k, class-balanced η² (XJTU-SY) — options for later (D263).
**Consequences:** `studies/spectral_probe/band_eta2.py`, `results/<dataset>_eta2.csv`, a new section in
`spectral_probe.ipynb`. XJTU-SY when its runs exist.

## D271 — Spectral probe notebook: band prediction only; spectra notebook (2026-11-18)
**Status:** Accepted
**Context:** The user wants to look only at how well the bands are predicted from the SSL representations
and the FFT, not at class prediction accuracy, and to judge the bands by eye on the datasets' spectra.
**Decision:** `spectral_probe.ipynb` keeps the band R² figures and the η² section (D270); premise tests (a)
and (c) (D245, D252) are no longer shown and premise test (b) (D245), which lived only in the notebook, is
removed. New `band_spectra.ipynb`: per dataset, class median spectra of the windows as the encoders received
them, the targets per class, and single windows, with the probe bands and their η² overlaid.
**Alternatives:** keeping the premise tests in the notebook (not wanted now; their scripts and results stay).
**Consequences:** README of the study updated; (b) can be restored from git history.
