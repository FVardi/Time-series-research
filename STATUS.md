# STATUS — what is done, what is ready, what is open

Living list of the state of the repository. Update it when something changes;
decisions themselves stay in `decisions/` (index: `DECISIONS.md`), questions for the
supervisor in `QUESTIONS.md`. Last updated: 2026-10-27 (evening).

## 1. FordA benchmark (frozen probes, D26: `svm_ts2vec` + `logreg`, 1/10/100 % labels, 3 seeds)

| method | state | run | reproduction (SVM, 100 %) |
|---|---|---|---|
| TS2Vec | done | `runs/20261020_144711_ts2vec_forda` | 0.936 vs 0.936: pass |
| T-Loss | done | `runs/20261020_144932_tloss_forda` | 0.932 vs 0.928: pass |
| TS-TCC | done (2026-10-27) | `runs/20261027_151321_tstcc_forda` | 0.931 vs 0.930: pass |
| TS-JEPA (masking fixed) | done, not timed (D102) | `runs/20261016_114616_tsjepa_forda_fixed` | check on the official-masking run (D103) |
| PatchTST | done | `runs/20261019_154305_patchtst_forda_flatten` (also `_mean`) | nothing published |
| TF-C (positives, D122–D124) | done (2026-10-27) | `runs/20261027_144405_tfc_forda` | nothing published |
| TF-JEPA (single pass, D113) | done (2026-10-27) | `runs/20261027_153543_tfjepa_forda` | nothing published |
| SimMTM | done (2026-10-27) | `runs/20261027_153657_simmtm_forda` | nothing published |
| FreRA | done (2026-10-27) | `runs/20261027_153909_frera_forda` | open: FordA value in the paper's Table 8 not yet read (paper protocol: best test accuracy) |
| raw, FFT, catch22, MiniRocket | done | `runs/20261020_14*_{raw,fft,catch22}_forda`, `runs/20261020_152809_minirocket_forda` | — |

Run: `uv run python scripts/pretrain.py configs/<method>_forda.yaml`, then the `evaluate.py` command it prints.

## 2. Methods: planned or in progress

- **Random-init frozen encoders** (D134; probes on untrained encoders, to see how much pretraining helps):
  implemented; FordA configs ready, **to run**: 9 × `configs/*_random.yaml` (a) and 6 × `*_random_bn.yaml` (b).
  CWRU/Paderborn copies follow when those trained configs exist (D134). Earlier: TF-C on FD (`scripts/probe_tfc.py`).
- **CA-TCC**: deferred together with pseudo-labelling (D104).
- **Further methods**: to be chosen.
- *(Optional)* frozen probes on the TF-C (positives) FD-A → FD-B encoders (`runs/20261027_141808_tfc_positives_fd`;
  copy of `configs/probe_tfc_fd.yaml` pointing at that run).

## 3. Datasets

| dataset | state | open |
|---|---|---|
| FordA | converted, in use (D27, D30, D31) | — |
| FD-A / FD-B | converted; TF-C reproduction only | — |
| CWRU | raw files downloaded; split settled (D128), probe CV grouped by bearing (D129); **not converted** (no `convert_cwru.py`) | sampling-rate matching of the normal files (D126); window length and input representation; one config per method |
| Paderborn | raw archives downloaded (D47); signals chosen (D48); **not converted** | conversion, task, split, windows |
| Herning | conversion code tested; **not fully converted**; status `validation` (rig unfinished) | κ inputs (D12); split (Q1); sweep vs. window comparison (Q2); VFD comb (Q3); high-pass filtering; final sampling rate |
| Jim-Tec | EDA notebook only (`eda/jim-tec/`) | everything (no decisions file yet) |
| FordB | not adopted | Q4 |
| run-to-failure / RUL | not adopted | D115 |

Standing rule (D123): every new benchmark dataset gets a TF-C (positives) config with the same protocol as the other methods.

## 4. Loose ends

- **HANDOFF.md**: superseded by this file (D130); to be removed with `git rm HANDOFF.md`.
- **Commit and submodules**: much of `runs/` was made from uncommitted trees (D99); `external/frera`, `external/simmtm`, `external/tfjepa` were added as submodules on 2026-10-27 and need committing.
- **Loss logs**: TS2Vec and T-Loss now log the loss per iteration/step (D143); FordA runs repeated: `runs/20261027_164037_ts2vec_forda` (600 values), `runs/20261027_164446_tloss_forda` (2,000 values).
  Longer-training check (D142) done: `runs/20261027_172356_tfjepa_forda_40ep`, `runs/20261027_172704_simmtm_forda_20ep`, `runs/20261027_174528_tfc_forda_80ep`; judged (D145): standard settings kept, these runs are checks only.
- **`tfc.scores`**: AUROC fails on binary tasks (two-column probabilities), so TF-C fine-tuning (`scripts/tfc_ours.py`) cannot run on two-class datasets.
- **TF-C study**: ablation stages have seed 0 only (D96: up to 3); `collapse_fd.yaml` has two placeholders (seeds, diagnostic windows); collapse examination of the official TF-C code; longer-training check: see D142 above.
- **TS-JEPA**: rerun with timing (D102).

## 5. Deferred by the user

- Statistics across methods (critical-difference diagrams etc.).
- A common fine-tuning protocol for all methods.
- Results report (`report/results.tex`): updated only when asked.
