# Time series research: checklist (archived 2026-11-17)

Retired 2026-11-17: the user keeps their own checklist in `report/`. This is the last state of
the former claude.ai Project doc `claude/CHECKLIST.md`, kept as a reference for open items. It is
no longer updated. Decision numbers D1–D269 refer to `decisions_archive/`.

**Standard runs (D239):** one run per method and dataset, `configs/<method>_<dataset>.yaml`, the dataset's one split (D216), seed 0. Only these go into benchmark tables; superseded runs live in `runs/archive/`; diagnostics keep suffixed names. FordA keeps its probes without `floor` (user's decision, 2026-11-10).

**Normalisation standard (D253, D255):** method inputs and baselines standardised per channel with the pretraining data's statistics (D137) on every dataset; no per-instance normalisation (documented departures: CWRU/Zhao, XJTU-SY/DSCN); FordA as distributed.

**Benchmark baselines (D267):** |FFT| (not Herning), squared envelope spectrum (bearing data), MiniRocket, plus the floor; Herning keeps MSSP statistics and pooled band features. catch24 and raw samples are no longer benchmark baselines, and catch24 is out of the spectral probe study too (D268).

Last updated: 2026-11-17 (10:30, restructuring chat: D269, EDA `.tex` files and `report/results.tex` deleted; before that 10:25, run-overview chat)

## Main tasks for the week of 2026-11-09
### A. Can SSL representations predict the dominant parts of the original spectrum? (`studies/spectral_probe/`)
- [x] Setup decided (D244): 16 equal-width bands; targets log absolute / relative band energy, log total energy, centroid (relative + centroid only on FordA and XJTU-SY); ridge per target on the stored Z; FFT, catch24, MiniRocket and random-init references (catch24 dropped, D268); FordA, CWRU, HUST, XJTU-SY, Paderborn (10 combinations); Herning left out
- [x] Changed (D246): all training windows only (no 1/10 %), alpha grid 10⁻⁶…10⁸; probes run again
- [x] Premise tests decided (D245): (a) spectral targets as features of the benchmark logreg probe; (b) spectral R² against benchmark logreg accuracy (notebook, Spearman's ρ). Added (D252): (c) each band's relative energy alone as the logreg feature
- [x] Probes, (a) and (c) run: `results/{forda,cwru,hust,paderborn}{,_classify,_bands}.csv`; `xjtu_sy.csv` (probe only)
- [ ] Rerun XJTU-SY once its runs are redone (D247, D254 archived the earlier runs), including (a) and (c) on the stage task. With D266 windows the stored Z has one row per 2,560-point window (the benchmark averages per minute at evaluation), so the probe works per window unchanged
- [x] Per-bearing measure for XJTU-SY added (D265): `per_bearing: true` in `configs/xjtu_sy.yaml` (false elsewhere) writes `results/xjtu_sy_per_bearing.csv`, Pearson's r between predictions and targets within each test bearing; first used with the XJTU-SY rerun
- [x] Premise test (b) in `spectral_probe.ipynb` (the notebook also has the per-band R² and (c) figures); first look 2026-11-10: on FordA spectral R² hardly ranks the methods (Spearman ρ 0.40 over 14 SSL runs); the bands lost (above 0.125 cycles/sample) carry almost no class information alone; random-init encoders hold as much spectrum as trained ones; the spectral summary alone classifies CWRU (1.00) and HUST (0.92) but not FordA (0.86)
- [x] Rows added to the configs (2026-11-12): FordA TS2Vec and T-Loss (D242, probed 2026-11-10); HUST TS2Vec, TS-TCC, SimMTM, PatchTST, FreRA, T-Loss, TS-JEPA (`configs/hust.yaml`, 13 runs); Paderborn `positives_lambda0` rerun, TS2Vec, TS-TCC, SimMTM, PatchTST, T-Loss (`configs/paderborn.yaml`, 110 runs, listed by combination)
- [x] Probes rerun on HUST and Paderborn with all finished methods (2026-11-12: `results/hust.csv` 15:37, `results/paderborn.csv` 18:25). Spectral R² (relative bands) against benchmark logreg accuracy, Spearman ρ over the SSL runs: HUST 0.95 (13 runs; 0.89 over the 7 non-TF-C methods), Paderborn ≈ 0 (11 runs; −0.30 over the 5 non-TF-C methods), FordA 0.40. On HUST the TF-C variants keep more spectrum than the FFT reference (0.90–0.95 vs 0.77) and classify best; the other methods keep less (0.42–0.76) and, except TS-TCC, TS2Vec and FreRA, classify below the spectral summary alone (0.92). On Paderborn the arms that keep most spectrum (TF-C λ = 0, 0.73–0.76) classify poorly (0.52–0.53); SimMTM is most accurate (0.67) with little spectrum (0.47)
- [x] Rows added (2026-11-14): Paderborn TS-JEPA (c0–c9, done 2026-11-13); MiniRocket as reference row on Paderborn and HUST. catch24 rows removed again from FordA, HUST and Paderborn (D268, 2026-11-17): `configs/forda.yaml` 30 runs, `configs/hust.yaml` 14, `configs/paderborn.yaml` 130 (13 per combination). FreRA skipped on Paderborn (D264). Config changes not committed yet
- [ ] **Rerun `probe_spectral.py` on `forda.yaml`, `hust.yaml` and `paderborn.yaml`** (CPU only; Paderborn about 40 min; FordA and HUST minutes): FordA to drop its catch24 rows (D268), HUST and Paderborn for the new rows. (a) and (c) need no rerun (same windows). Not run as of 2026-11-17 10:05 (`results/paderborn.csv` still from 2026-11-12)
- [ ] Add as they finish: CWRU standard runs (Next up 3, 4); XJTU-SY runs (D266 windows)
- [ ] Arrow figure, wanted once the runs are done (user, 2026-11-10): per dataset, spectral R² (x) against accuracy (y), one arrow per method from its random-init to its trained encoder, line at the spectral summary's accuracy (a). Needs random-init runs on the bearing datasets (D134: CWRU, Paderborn). Trajectories (a phase diagram proper) would need representations saved at checkpoints during training: not now
- [ ] Options, not built (D263: simple first, refinements one at a time when the user asks): R² weighted by band relevance as a second x-axis for (b); finer bands. Not used: FordA's longer-training runs as extra points (D263: extras for all methods or none)
- [ ] Not now (D244, D245): bands chosen per class; Herning

### B. Overview of time–frequency methods in SSL for time series — **on hold** (user, 2026-11-11; to be done in its own chat)
- [ ] Decide the scope and format (where the overview lives; what goes into it per method)
- [ ] Three families: contrastive, JEPA, masked reconstruction
- [ ] Compile the method list from the literature (harness methods with a frequency part as a starting point: TF-C, TF-JEPA, FreRA)
- [ ] For each method: how time and frequency enter (views, encoders, losses, targets)
- [ ] Write the overview

## Next up (suggested order)
1. [x] **Commit** `eb60840` (D256 configs), `9743a99` (queue script) (2026-11-10); restructuring `1b50d84` (2026-11-12 16:40); dataset descriptions `05b6942` (2026-11-16 22:29); D267 + `scripts/queue_d267.ps1` `b7fd315` (2026-11-17)
2. [x] **D256 queues done** (`scripts/queue_d256.ps1`; logs `runs/queue_d256_{gpu,cpu}.log`; no tracebacks):
   - [x] GPU, 2026-11-10 16:26 → 2026-11-13 03:48: `tfc_paderborn_positives_lambda0`; HUST TS2Vec, TS-TCC, SimMTM, PatchTST, FreRA, T-Loss, TS-JEPA; Paderborn TS2Vec, TS-TCC, SimMTM, PatchTST, T-Loss, TS-JEPA (`runs/20261113_034759_tsjepa_paderborn_combinations.csv`; combinations c6–c9 took ~4.5–5 h each instead of ~3 h). Paderborn FreRA skipped (OOM, D264)
   - [x] **Paderborn FreRA: CUDA out of memory** (2026-11-11 17:28). Cause: the official projector Linear(d, d) on the flattened FCN output, d = 82,176 on 5,120 samples (6.75 B weights). **Decided (D264): FreRA skipped on Paderborn and XJTU-SY, as TF-JEPA; partial folder archived; cards updated**
   - [x] CPU, 2026-11-13 03:48 → 06:10 (started after the GPU queue, on commit `1b50d84`): Paderborn FFT, envelope, catch22, MiniRocket (`runs/20261113_0*_{fft,envelope,catch22,minirocket}_paderborn_combinations.csv`); HUST FFT, envelope, catch22, MiniRocket (`runs/20261113_05*_<method>_hust`); XJTU-SY stage FFT, envelope, catch22, MiniRocket (`runs/20261113_05*_<method>_xjtu_sy_stage`, already with D266 windows: 2,560 points, mean per minute)
3. [ ] **Baseline queue `scripts/queue_d267.ps1` (CPU) ready to start** (user runs it, 2026-11-17; log `runs/queue_d267.log`): evaluates `runs/20261111_131844_mssp_herning` (the D255 MSSP rerun, pretrained 2026-11-11, never evaluated), then XJTU-SY RUL FFT, envelope, MiniRocket and CWRU FFT, envelope, MiniRocket
4. [ ] **Not yet in any queue** (the GPU has been idle since 2026-11-13 03:48):
   - [ ] XJTU-SY: all configs ready (D266) and already queued in `scripts/queue_xjtu.ps1` (in `1b50d84`, never started): `-Queue gpu` = TS2Vec, TF-C, T-Loss, TS-JEPA (173 epochs, D256 step rule), TS-TCC, SimMTM, PatchTST on RUL and stage, then the 10 TF-C variants (24 configs); `-Queue cpu` = RUL FFT, envelope, MiniRocket (overlaps queue_d267; finished configs are skipped). catch22 removed from it 2026-11-17 (D267; configs chat, uncommitted)
   - [ ] CWRU standard runs: the SSL methods at seed 0 (`{frera,patchtst,simmtm,tfjepa,tloss,ts2vec,tsjepa,tstcc}_cwru.yaml`, ready). Baselines in queue_d267; catch22 dropped (D267)
5. [ ] **Method benchmark per dataset** (`scripts/compare.py`, D203: hand-written `configs/compare/<dataset>.yaml` naming one standard run per method; writes `summary.csv`, `paired.csv` against the best baseline (D78, D80), `timing.csv`; in a task chat). Baselines per D267 (FFT, envelope, MiniRocket; Herning MSSP + pooled band features). Only `configs/compare/forda.yaml` exists (last run 2026-11-04):
   - [ ] FordA: ready now; rerun to pick up the D242 TS2Vec / T-Loss runs (catch24 and raw samples removed from `forda.yaml`, D267)
   - [ ] HUST: **ready now** (all SSL methods and baselines done, 2026-11-13)
   - [ ] Paderborn: **all runs done** (2026-11-13; FreRA skipped, D264). `compare.py` does not read the per-combination runs (`_c0`–`_c9`), so it needs extending, or Paderborn stays with its `_combinations.csv` means
   - [ ] CWRU: waits for the standard runs (Next up 3, 4)
   - [ ] XJTU-SY: waits for the redone runs (stage baselines done); Herning: waits for the MSSP evaluation (Next up 3)
6. [ ] **TF-C variant comparison → decide the benchmark variant (D122)** from D202 + D217 + D227 + D243. All arms exist on FordA, CWRU, HUST, Herning, Paderborn (side by side in `tfc_loss_variants_results.xlsx`, 2026-11-06; the last Paderborn arm, `positives_lambda0`, now complete); XJTU-SY to redo. Herning weighs less (validation data)

## TF-C study
- [x] TF-C loss comparison (D202), FordA, seed 0: done (`runs/20261104_13*_tfc_forda_{paper,left_out,left_out_x3}`); CWRU rerun on the Zhao protocol 2026-11-05
- [x] TF-C variants (D202, D217, D227, D243) run on FordA, CWRU, HUST, Herning and Paderborn (last Paderborn arm 2026-11-10). XJTU-SY runs archived (D247, D254) and to redo (Next up 4)
- [ ] Decide from D202 + D217 + D227 + D243 (Next up 6): paper (negatives) worst so far and collapsed (RankMe 4–8); λ = 0 as good or best on FordA, CWRU, HUST; Paderborn ranks left_out first; positives vs left-out arms not separated on one seed. Open: keep positives (D122), switch, or get more evidence first
- [ ] D217 open point: in arm (a) BatchNorm running statistics still see the augmented batches, so strict "no augmentation" would need a code change
- [ ] FD-A → FD-B transfer with all four TF-C variants (D211): pipeline, probes and seeds still to decide
- [ ] Later: extend the loss comparison to further datasets as they are converted
- [ ] Fix `tfc.scores` AUROC on binary tasks (blocks TF-C fine-tuning on two-class data)
- [ ] Collapse examination of the official TF-C code
- [ ] Fill the `collapse_fd.yaml` placeholders (seeds, diagnostic windows)
- [ ] More seeds for the ablation stages (D96)

## Standard runs (D239)
- [x] FordA: TS2Vec and T-Loss standard runs made again (D242): `runs/20261110_095303_ts2vec_forda`, `runs/20261110_095353_tloss_forda`
- [x] CWRU TF-C (positives) + variants on the Zhao protocol (2026-11-05)
- [ ] CWRU: the other 11 methods at seed 0 (catch22 dropped, D267): baselines in queue_d267 (Next up 3), SSL methods not queued (Next up 4)
- [x] HUST TF-C (positives): `runs/20261105_150730_tfc_hust` (2026-11-05)
- [x] HUST D256 runs: TS2Vec, TS-TCC, SimMTM, PatchTST, FreRA, T-Loss (2026-11-10), TS-JEPA (`runs/20261111_065149_tsjepa_hust`, 2026-11-11); baselines FFT, envelope, catch22, MiniRocket (2026-11-13)
- [ ] **PatchTST collapses** (noted, not investigated, by the user's decision): HUST standard run `runs/20261110_171649_patchtst_hust` has RankMe 3.7 of 128 and logreg 0.46 at 100 %; the masked-patch loss stays near 1.0, as on CWRU and Herning. Kept as the standard run
- [x] Paderborn TF-C (positives) (2026-11-05); TS2Vec (2026-11-10); TS-TCC, SimMTM, PatchTST, T-Loss (2026-11-11); TS-JEPA and baselines FFT, envelope, catch22, MiniRocket (2026-11-13). FreRA skipped (D264)
- [x] Configs for the other methods on Paderborn and HUST (FreRA, PatchTST, SimMTM, T-Loss, TS2Vec, TS-JEPA, TS-TCC) and baselines on HUST and XJTU-SY stage (D256, 22 files). TS-JEPA epochs = FordA's 565,000 steps, capped at 5,000 epochs (Paderborn 2,004, HUST 1,290); mixed precision tested, not used
- [ ] TF-JEPA on the new datasets: skipped for now (D256: one-token design, width = window length; ~1.7 B parameters on Paderborn). With HUST at 2,048 and XJTU-SY at 2,560 (D266) it may fit (rough scaling, not checked)
- [x] XJTU-SY SSL configs, RUL and stage (D266, 12 × 2,560-point windows, mean per minute): TS2Vec, T-Loss, TS-TCC (`features_len` 322), TS-JEPA (173 epochs), PatchTST, SimMTM (`CNNoutput_channel` 109); values checked by the configs chat 2026-11-12. FreRA (D264) and TF-JEPA (D256) skipped. Runs: Next up 4
- [ ] Random-init frozen encoders: CWRU and Paderborn copies (D134). The FordA runs are done (2026-10-27)

## Datasets
Open dataset setups (UOS, Ottawa, Paderborn RTF) are worked out in their own task chats; decisions are recorded there.
- [x] **Dataset descriptions** (commit `05b6942`, 2026-11-16): one LaTeX description and one EDA notebook per dataset in `eda/<dataset>/` (CWRU, DCASE 2022, FD, FordA, Herning, HUST, Jim-Tec, Ottawa, Paderborn, Paderborn RTF, UOS, XJTU-SY; PDFs built for CWRU, DCASE, FordA, HUST, Jim-Tec, Paderborn); `eda/README.md` updated. Not recorded on the checklist by the chat that made them (caught up 2026-11-17). The LaTeX descriptions and their PDFs/build files were deleted again (D269, 2026-11-17); the notebooks stay
- [x] CWRU: one split, the Zhao et al. protocol (D218); configs renamed `*_cwru.yaml`, earlier runs archived (D240)
- [x] Paderborn: converted (D212); split = Lessmeier et al. Section 7.3, 10 combinations (D220); probes/test setting 0, vibration, 5,120-sample windows (D222); pretraining on each combination's training windows (D225, supersedes D221)
- [x] HUST (Hanoi): converted (D223); task/split = the authors' 7-class experiment, stratified 80/20 (D226)
- [x] XJTU-SY: converted (D224); RUL target as DSCN (D228); FPT per Li et al. (2015) (D229, D230); DSCN split and metrics (D231); windows, input, tasks (D232)
- [x] XJTU-SY investigation (D247): no code error; D253 standardisation instead of DSCN's per-sample z-score (supersedes D232 item 3); ridge_grouped alpha grid 10⁻⁶…10⁸; baselines added on the RUL task; earlier runs archived. Windows: 12 × 2,560 points per minute, mean per minute (D266, supersedes D254 item 1)
- [ ] XJTU-SY: redo the runs (Next up 3, 4; stage baselines done 2026-11-13); health indicator task waits for the anomaly scorers (D192, D232)
- [ ] **Paderborn run-to-failure (`paderborn_rtf`, D234–D236):** data in `data/canonical/paderborn_rtf/`; next decisions: task, labels, split (authors' protocol as standard, D214; read github.com/alireza-javanmardi/bearing-RUL first: Tc value, which test cycles are scored) (task chat)
- [x] UOS converted (D248: run per file, 1,152 runs; channel `Data` as int32 codes, files with off-grid samples as float64; condition columns)
- [ ] UOS: task and split (open, no standard split: the user decides after the setup is explained; task chat)
- [x] Ottawa UODS-VAFDC converted (D249: run per bearing, stages 0–2 as captures, accelerometer + microphone at 42 kHz); task/split = Sehri et al. (2025): domains 2/4/6/8 train+val, 10 test, 5 classes (D250)
- [ ] Ottawa: train/val division, windows, inputs (open, D250; task chat)
- [x] DCASE 2022 bearing converted (D251: run per section, capture per clip, int16 PCM codes); official dev train/test split, AUC/pAUC
- [ ] DCASE: needs the anomaly scorers (D192)
- [ ] Anomaly-detection scorers (D192), needed for DCASE, XJTU-SY health indicator and others
- [ ] Jim-Tec: everything (no decisions yet)
- [ ] Each new dataset gets a TF-C positives config (D123)

## Herning
- [ ] Resume Herning work when the dataset is updated (MSSP baseline under D255: evaluation in queue_d267, Next up 3)

## Own method
- [ ] Split-representation pilot parked (D215: removal not working); next options in METHOD_IDEAS.md M1
- [x] Ideas logged (2026-11-12): METHOD_IDEAS.md M7 (spectral information in the objective), M8 (burst awareness: decomposition, pooling, augmentations, hit tokens), M9 (burst-reactive losses: higher-power / energy-weighted reconstruction, contrastive loss on supra-threshold parts), M10 (cheap tests). No decisions; not committed
- [ ] Possible cheap first test (M10): add burst targets (kurtosis, burst rate, burst energy fraction) to the spectral probe study

## Methods
- [ ] Choose further methods (task B may suggest candidates)
- [ ] FreRA: read the FordA value from Table 8 of the paper. Skipped on Paderborn and XJTU-SY (D264)
- [ ] PatchTST collapse (CWRU, Herning, HUST): not investigated for now (user's decision, 2026-11-10)

## Housekeeping
- [x] LaTeX build files in `eda/` (committed in `05b6942`): deleted with the `.tex` files (D269, 2026-11-17)
- [ ] Commit D269: deleted `eda/*/*.tex` + build files and `report/results.tex`; `README.md`, `eda/README.md`, `DECISIONS.md`, `decisions/repo.md` (uncommitted as of 2026-11-17)
- [ ] Left in place by D269, keep or delete (user): `scripts/make_report.py`, `report/tables/`, `report/figures/` (generated for the old `results.tex`)
- [x] Restructuring committed: `1b50d84` (2026-11-12 16:40; message "Dont know"), before the CPU queue started
- [x] `configs/tsjepa_paderborn.yaml` composed from its parts (D260); the merged config is identical to the file it ran with (configs chat, 2026-11-17; uncommitted)
- [ ] `Claude outputs/time_tsjepa_amp.py` (mixed-precision test, D256) and `Claude outputs/run_xjtu_first2560_test.ps1` (input-length test, D266): both committed in `1b50d84`; keep or delete (user)
- [ ] Reproduction check in `scripts/evaluate.py` always prints FAIL on single-seed runs: `max(2 * sd, 0.01)` with sd = NaN (one seed) is NaN in Python. TS2Vec / T-Loss on FordA are within 0.01 of the published values (0.9356 vs 0.936; 0.9364 vs 0.928). Fix not decided (spectral probe chat, 2026-11-10)
- [x] Run `scripts/status.py` to regenerate `STATUS.md` after the D256 queues (D261): done 2026-11-17 09:40 (run-overview chat); `STATUS.md` not committed yet
- [ ] Commit the spectral probe changes (`studies/spectral_probe/configs/{forda,hust,paderborn}.yaml`, `spectral_probe.ipynb`, `results/paderborn.csv`) and D268 (`decisions/evaluation.md`, `DECISIONS.md`, `docs/methods/baselines.md`); uncommitted as of 2026-11-17
- [ ] Stale git lock files moved aside under `.git/` (`stale-HEAD.lock-*`, `stale-maintenance.lock-*` and two older ones) because deleting in the repo folder was not permitted from the cloud shell; safe to delete by hand

## For the supervisor (QUESTIONS.md)
- [ ] Q1: Herning temperature hold-out (D56)
- [ ] Q2: whole-sweep features vs window-based SSL representations
- [ ] Q3: VFD interference / US band above 40 kHz
- [ ] Q4: add FordB?

## Deferred / on hold
- HUST (Huazhong), candidate C21 in `DATASET_CANDIDATES.md`: on hold (user, 2026-11-11). 9 health states, 11 speed conditions incl. one time-varying, 25.6 kHz; no licence stated; no standard split (would be our own, D216); needs a distinct name (e.g. `hust_huazhong`)
- Task B (time–frequency SSL overview): on hold, own chat (user, 2026-11-11)
- TS-JEPA epoch diagnostic on CWRU (D200): one 5,000-epoch run with RankMe + validation accuracy every 250 epochs; needs the GPU to itself
- Statistics across methods (CD diagrams etc.), until about 10 datasets are converted (D203)
- Common fine-tuning protocol
- Results document: written by the user in `report/` (D269); Claude does not write it
- Herning kurtogram (D186)

## Done / dropped
- [x] D269: EDA `.tex` dataset descriptions and their build files deleted; `report/results.tex` and build files deleted, the user writes the results document themselves (amends D98); `README.md`, `eda/README.md` updated (2026-11-17)
- [x] D268: catch24 also out of the spectral probe study (rows removed from the FordA, HUST and Paderborn configs; notebook baseline list; `docs/methods/baselines.md`; D267's spectral-probe sentence marked superseded) (2026-11-17)
- [x] D267: benchmark baselines = FFT, envelope, MiniRocket (+ floor; Herning keeps MSSP and pooled band features); catch24 and raw samples dropped from the benchmark (runs and configs kept; catch24 out of the spectral probe too, D268); `docs/methods/baselines.md`, `docs/protocol.md` P22, `configs/compare/forda.yaml` updated; queue `scripts/queue_d267.ps1`; commit `b7fd315` (2026-11-17)
- [x] Dataset descriptions, LaTeX + EDA notebook per dataset (`05b6942`, 2026-11-16)
- [x] D256 queues finished: GPU 2026-11-13 03:48 (Paderborn TS-JEPA last), CPU 06:10 (Paderborn, HUST and XJTU-SY stage baselines); no errors (2026-11-13)
- [x] Dedicated checklist chat retired (user, 2026-11-12); note added to D258 in `decisions/repo.md`
- [x] D265: per-bearing measure (Pearson's r within each test bearing) in the spectral probe for XJTU-SY (`per_bearing` config key) (2026-11-12)
- [x] D264: FreRA skipped where its official model does not fit (Paderborn, XJTU-SY); `runs/20261111_172727_frera_paderborn_c0` archived; `docs/methods/frera.md`, `docs/datasets/paderborn.md` updated (2026-11-12)
- [x] Repo restructuring (2026-11-12): collaboration rule D257 (standards taken without asking; the user decides what has no standard); current rules in `docs/` cards (D259); configs composed from `configs/datasets/` + `configs/methods/` parts (D260); `STATUS.md` generated by `scripts/status.py`, run it after runs (D261, supersedes D130); layout D262 (`harness/baselines/`, `harness/features/`, `studies/tfc_reproduction/`, `studies/method_checks/`, `eda/*/configs/`); root `README.md` and `studies/README.md`
- [x] D263: experiments start simple and grow one refinement at a time when asked; extras beyond the standard runs for all methods or none (`CLAUDE.md`, `docs/protocol.md` P28) (2026-11-12)
- [x] D258: every chat updates this checklist itself; rule in `CLAUDE.md` (bullet "Checklist (D258)") and in this header (2026-11-12)
- [x] D256 GPU queue: Paderborn T-Loss done (2026-11-11 17:27)
- [x] D256 GPU queue: Paderborn TS-TCC, SimMTM, PatchTST and HUST TS-JEPA done overnight (2026-11-11)
- [x] D256 GPU queue started 16:26; Paderborn TF-C arms complete, HUST 6 SSL methods, Paderborn TS2Vec (2026-11-10)
- [x] Commits `eb60840` (D256 configs), `9743a99` (queue script `scripts/queue_d256.ps1`) (2026-11-10)
- [x] Partial `tfc_paderborn_positives_lambda0` runs and the raw-input FFT Paderborn run moved to `runs/archive/` (2026-11-10)
- [x] D256: 22 configs (Paderborn and HUST, 7 SSL each; baselines for HUST and XJTU-SY stage) (2026-11-10)
- [x] UOS, Ottawa, DCASE converted (D248–D251); Paderborn RTF data converted (2026-11-10)
- [x] Normalisation standard for methods and baselines (D253, D255) (2026-11-10)
- [x] XJTU-SY investigated and fixed (D247, D254) (2026-11-10)
- [x] Spectral probe study decided, implemented and run (D244–D246, D252) (2026-11-10)
- [x] Commits `0444b92`, `9e44098`, `d482e04`, `dbecb8d` (2026-11-10)
- [x] FordA TS2Vec and T-Loss standard runs (D242) (2026-11-10)
- [x] Overnight TF-C variant queue, 35 of 36 arms (2026-11-05)
- [x] D243: TF-C variants on HUST, XJTU-SY, Herning; FordA λ = 0 configs (2026-11-05)
- [x] Standard-run rule (D239); CWRU configs renamed and old runs archived, D219 main-split runs cancelled (D240); FordA/Herning standard configs (D241) (2026-11-05)
- [x] Repo: FreRA and TF-JEPA save only the weights encode() uses (D237); LF line endings (D238) (2026-11-05)
- [x] Dropped: LightGBM baseline (D233) (2026-11-05)
- [x] Paderborn RTF added (D234), conversion decided (D235, D236) (2026-11-05)
- [x] D227: TF-C variants on Paderborn (2026-11-05)
- [x] HUST converted and split (D223, D226); Paderborn protocol (D220, D222, D225) (2026-11-05)
- [x] Split rules: standard literature split, one per dataset (D214, D216)
- [x] Datasets adopted and downloaded (D189–D194); raw data kept unpacked (D213)
- [x] Commits `6648159` and `64212f3` (2026-11-05)
- [x] D218 CWRU split = Zhao protocol; D217 TF-C consistency term only (2026-11-04)
- [x] Paderborn converted (D212); number clash resolved
- [x] Split-representation pilot synthetic runs (D204, D206, D207); pilot parked (D215)
- [x] Random-init FordA encoders, all 15 runs (2026-10-27)
- [x] Decision log updated (2026-11-04): D208 CA-TCC dropped; D209 no MLP probe in the pilot; D210 TS-JEPA FordA rerun dropped; D211 FD transfer with all four TF-C variants
