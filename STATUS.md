# STATUS — what is done, what is ready, what is open

Living list of the state of the repository. Update it when something changes;
decisions themselves stay in `decisions/` (index: `DECISIONS.md`), questions for the
supervisor in `QUESTIONS.md`. Last updated: 2026-11-05 (D239–D241: one standard run per method and dataset; section 3 as of 2026-11-05).

**Standard runs (D239):** per dataset, one run per method from `configs/<method>_<dataset>.yaml`, on the dataset's one split (D216), seed 0.
Only these go into comparison tables. Superseded runs and runs on other splits are in `runs/archive/`; diagnostics keep
suffixed names (`_random`, `_random_bn`, `_40ep`, `_flatten`, `_official_masking`, …) and stay out of the tables.

## 1. FordA benchmark (frozen probes, D26: `svm_ts2vec` + `logreg`, 1/10/100 % labels, 3 seeds)

| method | state | run | reproduction (SVM, 100 %) |
|---|---|---|---|
| TS2Vec | **to rerun** (D242): `configs/ts2vec_forda.yaml`, seed 0; earlier runs in `runs/archive/` | — | earlier: 0.936 vs 0.936: pass |
| T-Loss | **to rerun** (D242): `configs/tloss_forda.yaml`, seed 0; earlier runs in `runs/archive/` | — | earlier: 0.932 vs 0.928: pass |
| TS-TCC | done (2026-10-27) | `runs/20261027_151321_tstcc_forda` | 0.931 vs 0.930: pass |
| TS-JEPA (masking fixed) | done, not timed (D102); config now `configs/tsjepa_forda.yaml` (D241) | `runs/20261016_114616_tsjepa_forda_fixed` | check on the official-masking run (D103) |
| PatchTST | done (mean pooling, D87) | `runs/20261019_154248_patchtst_forda_mean` (`_flatten`: diagnostic, D86) | nothing published |
| TF-C (positives, D122–D124) | done (2026-10-27) | `runs/20261027_144405_tfc_forda` | nothing published |
| TF-JEPA (single pass, D113) | done (2026-10-27) | `runs/20261027_153543_tfjepa_forda` | nothing published |
| SimMTM | done (2026-10-27) | `runs/20261027_153657_simmtm_forda` | nothing published |
| FreRA | done (2026-10-27) | `runs/20261027_153909_frera_forda` | open: FordA value in the paper's Table 8 not yet read (paper protocol: best test accuracy) |
| raw, FFT, catch22, MiniRocket | done | `runs/20261020_14*_{raw,fft,catch22}_forda`, `runs/20261020_152809_minirocket_forda` | — |

Run: `uv run python scripts/pretrain.py configs/<method>_forda.yaml`, then the `evaluate.py` command it prints.

## 2. Methods: planned or in progress

- **TF-C loss comparison** (D202; pairs as positives vs negatives vs left out, seed 0) and consistency-only arms (D217):
  FordA runs done 2026-11-04 (`runs/20261104_13*_tfc_forda_{paper,left_out,left_out_x3}`); FordA λ = 0 arms not configured yet.
  CWRU: configs `configs/tfc_cwru_{paper,left_out,left_out_x3,left_out_lambda0,positives_lambda0}.yaml` on the protocol (D240),
  **to run** (the 2026-11-04 main-split runs are archived). Paderborn configured (D227).
  FD-A → FD-B transfer with all four variants (D211): pipeline, probes, seeds open.
- **Random-init frozen encoders** (D134; probes on untrained encoders, to see how much pretraining helps):
  FordA done (2026-10-27, 15 runs); CWRU and Paderborn copies still to make (D134); they are diagnostics, outside the benchmark tables (D239). Earlier: TF-C on FD (`scripts/probe_tfc.py`).
- **CA-TCC**: dropped (D208).
- **LightGBM**: dropped (D233).
- **Further methods**: to be chosen.
- *(Optional)* frozen probes on the TF-C (positives) FD-A → FD-B encoders (`runs/20261027_141808_tfc_positives_fd`;
  copy of `configs/probe_tfc_fd.yaml` pointing at that run).

## 2b. Herning pipeline (validation run; paused until the final dataset)

Everything below ran on the current validation run (one rig run, one bearing, one lubricant). The user waits with
further Herning runs until the dataset is final; then: convert, `scripts/derive_features.py` for `band`, `mssp`
(and `subbands`), and rerun the configs.

- **Pipeline:** κ labels from the telemetry (D55; `harness/labels.py`, `task: regression`); random 80/20 split of the
  sweeps with κ, one fixed draw (D169, validation only); ridge probe with ungrouped, shuffled folds, MAE and R² (D170).
  SSL input: per-window band features, 200 windows of 1 ms × 48 per sweep (D165), stored once (D172).
- **Baselines (done):** pooled band features, mean + SD over windows (D166, D187): `runs/20261102_112102_pooled_herning`;
  MSSP statistics per band and broadband (D181): `runs/20261102_103532_mssp_herning`. No catch24 on Herning (D188).
  The superseded mean-only version is `runs/archive/20261030_153010_pooled_herning`.
- **SSL methods** (configs `configs/<method>_herning.yaml`, D173): done for TS2Vec, T-Loss, TS-TCC, SimMTM, FreRA, TF-C,
  PatchTST, TF-JEPA (`runs/2026110[23]_*_<method>_herning`). TS-JEPA: patching fixed (D195); `runs/20261103_111018_tsjepa_herning`
  incomplete when this was written (seed 0 only); the failed starts `runs/20261103_104546_…` and `runs/20261103_110408_…`
  were deleted (2026-11-04).
- **Results so far** (random split, MAE of κ at 1 / 10 / 100 % labels): pooled 0.097 / 0.046 / 0.038; TS-TCC 0.087 / 0.046 / 0.041;
  MSSP 0.142 / 0.068 / 0.042; all other SSL methods below the pooled baseline. Caveat: κ is a function of rpm and temperature
  on this rig and the random split puts every operating point on both sides, so these mainly measure recognising the
  operating point (QUESTIONS.md Q1, update 2026-10-30). Within a speed κ varies only with temperature, so no split of this
  run separates lubrication from other temperature or time effects.
- **Open:** temperature hold-out (D56; options in QUESTIONS.md Q1); probe folds grouped by hold (D59; MSSP `grouping.py`
  not ported, D170 deviates); TS-JEPA on the final data.
- **EDA:** `eda/herning/envelope_model.ipynb` (envelope model, VFD-off reference, burst rate vs κ, scale scan, same-shape
  tests, burst durations); findings informed D165. Not in QUESTIONS.md yet: AE 50–100 kHz bursts at standstill disappear
  with the VFD off (converter interference; relevant to Q3).

## 3. Datasets

| dataset | state | open |
|---|---|---|
| FordA | converted, in use (D27, D30, D31) | — |
| FD-A / FD-B | converted; TF-C reproduction only | — |
| CWRU | converted (D139; 161 runs in `data/canonical/cwru/`); one split since D218: Zhao et al. protocol (D167); configs `*_cwru.yaml` (D240); earlier main- and bearing-split runs in `runs/archive/` | standard runs: none made yet, all 13 methods + TF-C variants **to run** (seed 0); D134 random-init copies |
| Paderborn | converted (D212); protocol D220, D222, D225 implemented (`harness/protocols.py`, `scripts/run_combinations.py`; ~9,000 training / 6,000 test windows per combination); baseline configs `{fft,envelope,catch22,minirocket}_paderborn.yaml`, not yet run | SSL configs: window-length-dependent settings (5,120 samples) per method |
| Herning | converted (all 3,964 sweeps, `data/canonical/herning/`); status `validation` (rig unfinished, final dataset pending; see 2b); derived files in `data/derived/herning/`: `band_features.h5` (SSL input, D165/D172), `mssp_stats.h5` (MSSP baseline, D181), `subband_power.h5` (1/12-octave sub-band power, D186; stored, not used yet) | κ inputs (D12); split (Q1); sweep vs. window comparison (Q2); VFD comb (Q3); high-pass filtering; final sampling rate |
| Jim-Tec | EDA notebook only (`eda/jim-tec/`) | everything (no decisions file yet) |
| FordB | not adopted | Q4 |
| Synthetic (D196) | generated (3 designs, 144 runs each); split-representation pilot run and parked (D215; METHOD_IDEAS M1) | — |
| XJTU-SY | adopted (D189); tasks and labels D191; raw downloaded (D193); not converted | FPT details (D191); split, windows |
| HUST | adopted (D189, D190); raw downloaded (D194); converted (D223; 99 runs in `data/canonical/hust/`); task/split D226 (authors' 7-class protocol: 2,048-sample windows, 75 % overlap, 2,500 per class, random 80/20); `configs/tfc_hust.yaml` (D123) | configs for the other methods |
| UOS | adopted (D189, D190); subsets 1–3 downloaded (D194); not converted | configuration |
| Ottawa UODS-VAFDC | adopted (D189); raw downloaded (D194); not converted | configuration |
| DCASE 2022 bearing | adopted (D189); protocol D192; dev set downloaded (D194); not converted | configuration; anomaly scorers not implemented |
| other run-to-failure / RUL | not adopted (D115, D189) | candidates in `DATASET_CANDIDATES.md` |

Standing rule (D123): every new benchmark dataset gets a TF-C (positives) config with the same protocol as the other methods.

## 4. Loose ends

- **Commits**: runs from 2026-10-30 to 2026-11-04 were made from uncommitted trees (D99; `git.json` in each run says so); the code was committed on 2026-11-04/05. HANDOFF.md removed and submodules committed.
- **Loss logs**: TS2Vec and T-Loss now log the loss per iteration/step (D143); the FordA reruns with loss logs are now in `runs/archive/` (D242).
  Longer-training check (D142) done: `runs/20261027_172356_tfjepa_forda_40ep`, `runs/20261027_172704_simmtm_forda_20ep`, `runs/20261027_174528_tfc_forda_80ep`; judged (D145): standard settings kept, these runs are checks only.
- **`tfc.scores`**: AUROC fails on binary tasks (two-column probabilities), so TF-C fine-tuning (`scripts/tfc_ours.py`) cannot run on two-class datasets.
- **TF-C study**: ablation stages have seed 0 only (D96: up to 3); `collapse_fd.yaml` has two placeholders (seeds, diagnostic windows); collapse examination of the official TF-C code; longer-training check: see D142 above.
- **TS-JEPA**: FordA rerun with timing dropped (D210).

## 5. Deferred by the user

- Statistics across methods (critical-difference diagrams etc.): deferred until about ten datasets are converted (D203). Comparison tables across runs exist: `uv run python scripts/compare.py configs/compare/<name>.yaml` (D203; first: `forda.yaml`).
- A common fine-tuning protocol for all methods.
- Results report (`report/results.tex`): updated only when asked.
- One-off kurtogram across the Herning data, to check the bands used for kurtosis (D186).
