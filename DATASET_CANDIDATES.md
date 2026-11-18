# Dataset candidates

Running log of datasets that could be used to test the methods, beyond the ones in use.
Claude's suggestions are listed here; they are not decisions. When a dataset is adopted,
record it in `DECISIONS.md` (D-number) and mark the entry here as "Adopted as Dn";
entries are not deleted. Details below are as stated on the source pages (checked
2026-11-02); anything not verified is marked as such.

In use or adopted before this log: FordA (D27), FD-A/FD-B (TF-C reproduction only, D39, D70),
CWRU (D115), Paderborn (D47, D70), Herning, Jim-Tec.

**Overlap warning:** FD-A/FD-B (TF-C's fault-diagnosis data) are derived from the Paderborn
data (Lessmeier et al., 2016). Pretraining on one and evaluating on the other may put the
same recordings on both sides.

---

## Unseen bearings and operating conditions (fault diagnosis)

### C1 — HUST bearing (Hanoi University of Science and Technology) — *Adopted as D189, D190*
- 5 ball-bearing types, 3 working conditions (loads/speeds not given on the page); 51.2 kHz; 99 files.
- Faults: inner, outer, ball crack and their two-defect combinations. CC BY 4.0.
- Why: test on a bearing type not seen in training. Sensor type/placement not stated on the page.
- https://data.mendeley.com/datasets/cbv7jyx4p9/3 (paper: BMC Research Notes, doi 10.1186/s13104-023-06400-4)

### C2 — UOS multi-domain vibration dataset (University of Seoul, 2024) — *Adopted as D189, D190*
- Three subsets, one bearing type each (subset 1: deep-groove 6204; other two not checked).
- Six speeds 600–1,600 RPM; 8 kHz and 16 kHz; single faults, misalignment, unbalance, looseness, 21 compound faults. CC BY 4.0.
- Sensors and load not stated on the subset-1 page.
- https://data.mendeley.com/datasets/53vtnjy6c6 ; paper doi 10.1016/j.dib.2024.110940

### C3 — University of Ottawa UODS-VAFDC (2023) — *Adopted as D189*
- 20 bearings, each recorded healthy (0), fault developing (1), faulty (2); classes H, I, O, B, C.
- Accelerometer + microphone (+ load, speed), 42 kHz, 10 s per recording, 1,750 RPM, ~400 N (ball faults: no load). CC BY 4.0.
- Why: early-stage detection; vibration and acoustic at once.
- https://data.mendeley.com/datasets/y2px5tg92h/1 ; https://www.sciencedirect.com/science/article/pii/S2352340923004456

### C4 — University of Ottawa, time-varying speed (2018)
- Vibration + encoder, 200 kHz, four time-varying speed profiles.
- Why: speed non-stationarity (order tracking).
- https://data.mendeley.com/datasets/v43hmbwxpm/1

### C5 — Politecnico di Torino (DIRG) rolling bearing rig
- Two triaxial accelerometers; oil-injected lubrication; speed 0–500 Hz, load 0–1,800 N; roller or inner-ring indentations 0/150/250/450 µm; plus a ~230 h endurance test. 3.2 GB.
- Sampling rate not stated on the Zenodo page.
- https://zenodo.org/records/3559553

### C6 — KAIST vibration/acoustic/temperature/current (2022)
- 4 accelerometers, 1 microphone, 2 thermocouples, 3-phase current; 25.6 and 50 kHz; 3 load conditions.
- Inner/outer race faults (3 severities), misalignment, unbalance, normal. CC BY 4.0.
- Why: multimodal input.
- https://data.mendeley.com/datasets/ztmf3m7h5x/3

### C7 — Common literature sets (low priority): JNU, SEU, MFPT, HIT-SM, DLR, SCA (pulp mill)
- Few bearings and/or fixed conditions; mainly for comparability with published numbers.
- Collection with links: https://github.com/sothulthorn/bearing-dataset

## Run-to-failure / natural degradation

### C8 — XJTU-SY — *Adopted as D189*
- 15 bearings, 5 per condition, 3 speed/load conditions; horizontal + vertical vibration, 25.6 kHz, 1.28 s every minute.
- Natural (accelerated) degradation; split by bearing possible.
- https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets

### C9 — FEMTO-ST / PRONOSTIA (IEEE PHM 2012)
- 17 bearings, 3 conditions; vibration 25.6 kHz, temperature.
- https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset (unofficial mirror)

### C10 — NASA IMS
- 3 experiments, 4 bearings each; 20 kHz; constant 2,000 RPM, 6,000 lbs.
- https://data.nasa.gov/dataset/ims-bearings

### C11 — KAIST run-to-failure (2024)
- One NSK 6205 bearing, 128 h; vibration x/y + bearing and ambient temperature, 25.6 kHz. CC BY 4.0.
- https://data.mendeley.com/datasets/5hcdd3tdvb/6

### C12 — Doseděl, Hnidka, Havránek: multimodal run-to-failure dataset (Brno)
- Not verified (pages could not be opened): sensors, access and licence unknown.
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7576713 ; https://ieee-dataport.org/documents/open-multimodal-dataset-rolling-bearing-degradation

### C13 — UNSW run-to-failure; University of Ferrara run-to-failure
- UNSW: 4 experiments, vibration + encoder, multiple speeds: https://data.mendeley.com/datasets/h4df4mgrfb/3
- Ferrara: 6 tests, constant speed and load: https://sfera.unife.it/handle/11392/2569668

## Acoustic (microphone)

### C14 — DCASE 2022 Task 2, machine type "bearing" — *Adopted as D189*
- Anomaly detection trained on normal sounds only; domain shifts: rotation velocity (section 00), microphone location (01), factory noise (02).
- 10 s single-channel clips; dev 990 source + 10 target normal clips per section for training, 100 normal + 100 anomalous per section for test.
- Dev https://zenodo.org/record/6355122 ; additional training https://zenodo.org/record/6462969 ; eval https://zenodo.org/record/6586456
- Task page: https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

### C15 — Multi-condition acoustic dataset of ball bearings (Guilin, Feb 2026)
- 3 synchronised microphones, 32,768 Hz; 3 speeds × 3 loads; 50 min per condition. CC BY 4.0.
- Proprietary B&K `.bkc` format; fault description on the page is unclear.
- https://data.mendeley.com/datasets/jvv7p4wfgv/1

### C16 — MIMII / ToyADMOS / other DCASE machine types
- Not bearings; machine-sound anomaly detection under domain shift.

## Lubrication and AE

No open rolling-bearing dataset with labelled lubrication states or MHz-rate AE was found
(2026-11-02). AE papers checked keep their data private.

### C17 — SoliDAIR journal bearings (Zenodo, 2025)
- Journal bearing (not rolling); label per data point: mixed lubrication (1) or not (0), over speed and load. 46 MB CSV; CC BY-SA 4.0. Sensors not documented on the page.
- Why: a lubrication-regime label, loosely analogous to κ.
- https://zenodo.org/records/17530066

### C18 — Rolling contact fatigue of AISI 52100 balls under calcium grease (Scientific Data, 2026)
- Not verified (page could not be opened).
- https://www.nature.com/articles/s41597-026-08057-7

## Generic time-series benchmarks

### C19 — UCR / UEA archives; TF-C's set (SleepEEG, Epilepsy, HAR, Gesture, EMG)
- Where the compared methods report results; one or two would make reproductions comparable.

---

## Added 2026-11-05 (second search)

Details as stated on the source pages (checked 2026-11-05); "not stated" means the page did not say.

### Unseen conditions / time-varying operation

#### C20 — Paderborn run-to-failure under time-varying load and speed (LDM, Paderborn University, 2024) — *Adopted as D234 (storage D235, conversion D236)*
- Not the Paderborn KAt fault data in use (D47): different rig, chair and recordings.
- Rig: shaft on two spherical roller bearings; static preload by lever and spring, dynamic load by an electrodynamic shaker. Test bearing 61806-2RS deep-groove ball bearing; no initiated defect (natural damage).
- 17 run-to-failure experiments (B01–B17). Snapshots of 1.6 s about every 12 s. Three one-directional accelerometers on the rig (A, C horizontal on the housing; B vertical on the lever); the data set provides A and C, the paper analyses only A. **Vibration at 128 kHz for B01–B09, 64 kHz from B10** (storage issues). Pt100 bearing and ambient temperature at 10 Hz.
- Operating conditions: speed constant within a snapshot, drawn uniformly per snapshot from a per-experiment interval (e.g. 2,400–3,000 rpm for B01–B03, 1,500–3,600 rpm for B06); static load e.g. 2,500–3,800 N; dynamic load sinusoidal at 2 Hz (B01–B06) or band-limited white noise up to 200 Hz (B07–B17), amplitude drawn per snapshot. OC vector: peak dynamic load, mean static load, mean speed.
- Stop: vibration RMS above 8 × mean RMS (about 6–10 g) or housing at 110 °C (temperature triggered for B03, B07, B08, B15).
- Lifetimes 377 (B01) to 22,567 (B15) snapshots, so very unbalanced across bearings.
- Size: 152 GB as 48 ZIP parts (B06, B13 and B15 about 30 GB each) plus a 2.8 MB description PDF. CC BY 4.0.
- **Authors' protocol** (Javanmardi et al. 2024): leave-one-bearing-out over all 17 bearings, results averaged; features = max FFT amplitude in 20 equal-width bands of accelerometer A; healthy/unhealthy transition from 2-means clustering on the training bearings (a minimum-cycle parameter Tc whose value in the main experiments is not stated); RUL target in % after the transition, trained on post-transition data only; MAE; GB and RF, 10 seeds. Which test cycles enter the MAE is not stated clearly. Code: https://github.com/alireza-javanmardi/bearing-RUL
- Why: natural degradation under non-stationary operation, split by bearing; close to D191's XJTU-SY tasks.
- Checked on B01 and the description PDF (v2, 2024-03-25), downloaded 2026-11-05:
  - Only accelerometers A and C are in the data set (variables `accHorizRear_A`, `accHorizFrontal_C`, `measTime`), in volts (float64); V-to-g factors in each experiment's `Bxx_log.pdf`. B01: 204,800 samples per channel per file (1.6 s at 128 kHz). MAT v5, readable with `scipy.io.loadmat`.
  - Accelerometers PCB 603C01: 100 mV/g, ±50 g, 0.5–10 kHz (3 dB), resonance 25 kHz.
  - Per experiment: `vibrationData/` (one `.mat` per measurement), `Bxx_operatingConditions.csv` (set and measured dynamic load, static load, speed per measurement), `Bxx_meanTemperatures.csv` (T1, T2, ambient), `Bxx_log.pdf`. The PDF also names `Bxx_extractedFeatures.csv`, which is not in B01.zip.
  - Files per experiment (Table 3): 377 (B01) to 22,567 (B15), 95,660 in total; durations 2–76 h. Defects found after dismantling per bearing (IR, OR, B, combinations).
  - No threshold criterion was pre-set for B01, B05 was interrupted deliberately, and B16 has no criterion listed: not all runs end at a defined failure.
  - B01 starts with a run-in: measurements 1–55 at 750 rpm, 2,500 N static load, no dynamic load, outside the Table 2 range. The measurement interval is not constant (about 12 s at the start, about 30 s around measurements 100–200 in B01).
  - B01.zip: 0.97 GB packed, 1.09 GB unpacked (ratio 1.13); the full set would be roughly 170 GB unpacked if that ratio holds.
  - Quantization (checked on B01 M0001 and M0377): both channels lie on a fixed grid of 16-bit ADC steps (A: 0.000323335 V, C: 0.000323356 V, each with its own offset; deviation from the grid below 1.2e-7 steps), and `measTime` is exactly regular. So the signals could be stored as int16 codes plus step and offset per channel, about 50 GB for all 17 experiments and both channels; not checked for B02–B17.
- https://zenodo.org/records/10868257 ; paper https://papers.phmsociety.org/index.php/phme/article/view/4101 (doi 10.36001/phme.2024.v8i1.4101)

#### C21 — HUST bearing (Huazhong University of Science and Technology; not the Hanoi HUST of C1)
- ER-16K bearing on a SpectraQuest simulator; 9 health states (normal; inner, outer, ball, inner+outer, each medium/severe); 11 speed conditions (10 constant, 20–80 Hz, plus 0-40-0 Hz time-varying); one accelerometer, 25.6 kHz, 262,144 points per file; 99 Excel files.
- **Licence: none stated** (the GitHub page only says the data are public). Download via Google Drive / Quark links.
- **No standard split found (checked 2026-11-05).** The Zhao, Zio & Shen (2024) DG benchmark (code MIT, https://github.com/CHAOZHAO-1/Domain-generalization-fault-diagnosis-benchmark) uses "M_HUST" only in its cross-machine tasks (as one of four machines, all conditions pooled), with 3 classes, 5 working conditions and 800 samples per class per condition, from preprocessed `.mat` files on the authors' Google Drive. Which HUST files/speeds those are could not be checked (supplementary PDF not reachable), so it is not certain it is this release; it is not a within-dataset protocol either. The cross-working-condition tasks use CWRU, not HUST.
- Naming clash with C1: needs a distinct name in the harness if adopted.
- https://github.com/CHAOZHAO-1/HUSTbearing-dataset

#### C22 — MCC5-THU gearbox under variable working conditions (Tsinghua, 2024)
- Two-stage parallel gearbox. Classes per the Data in Brief paper: health, missing teeth, gear wear, pitting, tooth crack, tooth break (light/medium/high), and compound faults **tooth break + bearing inner race** or **tooth break + bearing outer race** (light/medium/high; raceway fault widths 0.1/0.3/0.5 mm). **No bearing-only fault class.** (The GitHub README lists inner and outer race faults among the fault types; the paper only has them in compounds.)
- Two triaxial accelerometers (motor drive end, intermediate-shaft bearing seat), 12.8 kHz; speed and torque channels. 60 s per file, 240 CSV files.
- Conditions: time-varying speed at 10 or 20 Nm, or time-varying load (0–10 or 0–20 Nm) at 1,000/2,000/3,000 rpm; the paper says 12 working conditions but does not list them all, and has inconsistent counts (e.g. 48 vs 112 transitional combinations).
- **No official tasks or train/test splits:** the paper suggests building your own.
- Licence: CC BY 4.0 on Mendeley; MIT on GitHub. Use Mendeley version 2 (version 1 has reversed torque signs; torque needs a scaling factor of about 6, per the repo).
- https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets ; https://data.mendeley.com/datasets/p92gj2732w/2 ; paper doi 10.1016/j.dib.2024.110453

#### C23 — KAIST rolling-bearing vibration and motor current under varying speed (2023)
- Healthy, inner race, outer race; random varying speed 680–2,460 RPM; vibration x/y at two housings, 3-phase current, RPM; CSV. CC BY 4.0. Sampling rate and load not stated on the page.
- Split into Subset1–3 for storage. The page cites the same Data in Brief paper as C6, so it may belong to the same collection: check overlap before adopting.
- https://data.mendeley.com/datasets/j8d8pfkvj2/7 (Subset3)

#### C24 — BJTU-RAO bogie datasets (Beijing Jiaotong University, 2024/2025)
- Subway bogie transmission system (driving gearboxes, axle boxes, traction motors); 1 healthy + 50 faulty states, each under 9 working conditions; multi-sensor, 64 kHz.
- Download location not confirmed (released at the 2024 Global Reliability and PHM Conference). Sensor types and conditions not stated on the pages read.
- Paper: Qin et al., "An In-Depth Tutorial on BJTU-RAO Bogie Datasets for Fault Diagnosis", IEEE Access 2025, https://ieeexplore.ieee.org/document/10933494

### Multimodal

#### C25 — MaFaulDa (Machinery Fault Database, UFRJ)
- SpectraQuest rig, two bearings; 8 accelerometer channels (radial, axial, tangential), tachometer, microphone; 50 kHz, 5 s per sequence; 1,951 sequences. Classes: normal (49), imbalance (333), underhang bearing faults (558), overhang bearing faults (513), horizontal (197) and vertical (301) misalignment; normal at 737–3,686 RPM. Bearing-fault recordings carry added imbalance masses (6, 10 or 20 g).
- Figures as reported in arXiv 2301.02243 and 2412.01322; the official page blocks automated access. **Licence not found.** A gated Hugging Face copy exists (AI4Manufacturing/MAFAULDA-rotor-perception).
- **No standard split found (checked 2026-11-05).** Example: arXiv 2412.01322 uses 10 classes, segments of 48 rotations (length from the tachometer speed), stratified random 70/15/15 split of segments, so segments of one recording can end up on both sides; no standard protocol cited.
- https://www02.smt.ufrj.br/~offshore/mfs/page_01.html

### Lubrication

Still no open rolling-bearing dataset with labelled lubrication states (second search, 2026-11-05).

#### C26 — Oil-starvation run-to-failure with AE (Lubricants 12(6):229, 2024)
- One accelerated test: bearing given 2 mL of oil, run without relubrication until failure (90 °C or abnormal vibration); lubrication degradation detected after about 5,300 min.
- AE (Fujicera HS-10A-11M2) at 204.8 kHz; vibration (B&K 4535-B-001) and temperature at 25.6 kHz.
- Data not public: the availability statement says inquiries can go to the corresponding author. Closest match to Herning found so far (AE + lubrication degradation).
- https://www.mdpi.com/2075-4442/12/6/229

#### C27 — NREL Gearbox Reliability Collaborative vibration benchmarking datasets
- Two wind-turbine gearboxes of the same design: one healthy (dynamometer only), one damaged after two loss-of-oil events in the field (bearing and gear damage), retested on the dynamometer. Accelerometers + high-speed-shaft RPM.
- Sampling rate, size and download link not stated on the OSTI page; hosted through OEDI, terms in a separate licence document.
- Why: damage caused by lubrication loss.
- https://www.osti.gov/biblio/1844194 ; https://data.openei.org/files/738/

### Anomaly detection (normal-only training)

#### C28 — Airbus helicopter accelerometer dataset (Airbus SAS 2018, ETH Zurich)
- Helicopter accelerometers; 1,024 Hz; 1-min sequences (61,440 samples). Train: 1,677 normal sequences; test: 594 (297 normal, 297 anomalous); labels per flight. Normal-only training. Official train/test split.
- Not bearings specifically. Figures from Garcia et al. (arXiv 2005.07031).
- **Licence: CC BY-NC-SA 4.0** (rights holder Airbus SAS; checked on the ETH record 2026-11-05). Non-commercial and share-alike.
- https://doi.org/10.3929/ethz-b-000415151

#### C29 — DCASE 2023–2025 Task 2, machine type "bearing" (development sets)
- Bearing is in the development set in 2025 (10 or 12 s clips; 990 source + 10 target normal training clips; 100 normal + 100 anomalous test clips per section; no attributes for bearing).
- Not checked: whether these clips overlap with DCASE 2022 (C14).
- 2025 dev: https://zenodo.org/records/15097779 ; 2024 dev: https://zenodo.org/records/10902294

### Small or low-priority

- **C30 — FSTF bearing sound datasets** (Sidi Mohamed Ben Abdallah University, 2023): SKF 6004 on a Gunt PT 500.12 kit; normal, inner, outer, ball, looseness; several speeds (not given); stethoscope + smartphone or other microphone, 44.1 kHz WAV; CC BY 4.0. https://data.mendeley.com/datasets/n9y9c7xrz3/1
- **C31 — Lab-scale vibration dataset** (Atmaja et al.): normal, unbalance, misalignment, bearing fault; CSV; details not stated on the abstract page. https://zenodo.org/record/7006575 (paper arXiv 2212.14732)
- **C32 — University of Arkansas single and double faults** (Data in Brief 2023, doi 10.1016/j.dib.2023.109358): 39 single/compound fault scenarios plus healthy (per the awesome-bearing-dataset list; page not read).
- **C33 — Vishwakarma Institute of Technology**: 50 vibration files, healthy/inner/outer, loaded and unloaded. https://data.mendeley.com/datasets/zgzyxdnyv9/2
- **C34 — NUST ICE journal bearing** (engine journal bearings, triaxial vibration; doi 10.1016/j.dib.2024.111214) and **SUBF v2.0** bearing sounds (Kaggle). Not read.

### Not usable / collections

- **HUMS2025 Data Challenge** (Bell 206B-1 main rotor gearbox, 65.5 kHz, about 13.6 GB): participants only, and the fault is a casing crack and gear spalling, not bearings.
- **IEEE DataPort "Vibration signal datasets for bearing fault diagnosis and OOD detection"** (2025): subscription only; built from CWRU and a "PU" set.
- **PHM-Vibench** (PHM Society Asia-Pacific 2025; Apache 2.0 code): says 20+ datasets with standardized protocols, but the public data bundle currently covers CWRU only. Worth watching as a source of standard protocols. https://github.com/PHMbench/PHM-Vibench
- **Universal vibration dataset** (arXiv 2504.11581): pooled bearing data for transfer learning; sources and release not stated in the abstract.
- **awesome-bearing-dataset** (28 datasets with links): https://github.com/VictorBauler/awesome-bearing-dataset

---

## Added 2026-11-18 (third search): well-known non-bearing benchmarks

Asked for: well-known time series datasets outside bearings (EEG etc.) for testing the SSL
methods. Details as stated on the source pages (checked 2026-11-18); "not checked" means it
was not verified on a source page. Subject- or patient-wise splits are the nearest analogue
of the harness's unseen-bearing splits.

**Overlap warnings:** C35 SleepEEG is built from the Sleep-EDF sleep-cassette recordings
(C38). AdaTime's MFD data (C37) is reportedly derived from Paderborn (not checked), which
would overlap with D47 and FD-A/FD-B. Public BIOT checkpoints were pretrained on CHB-MIT,
TUAB and TUEV, so those sets are in-domain for BIOT.

### Where the compared SSL methods report results

#### C35 — TF-C fine-tuning sets: SleepEEG, Epilepsy, Gesture, EMG
- Preprocessed by the TF-C authors and put on figshare (same source as FD-A/FD-B, D39).
- SleepEEG: 153 whole-night EEG recordings from 82 healthy subjects (sleep cassette), 1 lead, 100 Hz. TF-C uses it for pretraining.
- Epilepsy: single-channel EEG from 500 subjects, 23.6 s each; TF-C merged the first 4 classes, giving a binary seizure label.
- Gesture: accelerometer, 8 gestures; only 440 measurements could be recovered via UCR (the original paper reports 4,480).
- EMG: single-channel, tibialis anterior, 4 kHz, from **three volunteers (healthy, neuropathy, myopathy)**, so class and subject coincide.
- Why: TF-C and follow-ups (e.g. SimMTM, FEI; not all rechecked) report on these, so they are the cheapest way to make harness numbers comparable with published ones. Train/val/test file layout not checked.
- https://figshare.com/articles/dataset/TF-C_Pretrain_SleepEEG/19930178 ; .../TF-C_Pretrain_Epilepsy/19930199 ; .../TF-C_Pretrain_Gesture/19930247 ; .../TF-C_Pretrain_EMG/19930250 ; code https://github.com/mims-harvard/TFC-pretraining

#### C36 — UCR (128 univariate) and UEA (30 multivariate) archives; Multiverse archive (2026)
- UCR expanded from 85 to 128 datasets in 2018; UEA has 30 multivariate classification datasets (2018). Each dataset has a fixed train/test split. FordA (D27) comes from UCR.
- Multiverse (Middlehurst et al., arXiv 2603.20352, March 2026): expands the UEA archive from 30 to 133 classification problems, or 147 including preprocessed versions of datasets with missing values or unequal lengths.
- Why: TS2Vec-style evaluations average over whole archives (not rechecked today). Mostly short, low-rate series, unlike bearings.
- https://www.timeseriesclassification.com ; https://arxiv.org/abs/2603.20352

#### C37 — AdaTime domain-adaptation benchmark (Ragab, Eldele et al., ACM TKDD 2023)
- Datasets: UCIHAR, WISDM, HHAR, sleep stage classification (SSC, from Sleep-EDF) and MFD (machine fault diagnosis). Each subject is a domain, with fixed source→target scenarios: 5 per dataset in arXiv v1, 10 in the version later papers use. Preprocessed data via the GitHub repo.
- Why: a ready-made, standard protocol for shift between subjects, which is the closest literature analogue to unseen bearings and conditions. Built for domain adaptation, but the fixed scenarios also work as cross-domain test splits for frozen SSL encoders.
- https://github.com/emadeldeen24/AdaTime ; https://arxiv.org/abs/2203.08321

### EEG

#### C38 — Sleep-EDF Database Expanded (PhysioNet)
- 197 whole-night PSGs: EEG (Fpz-Cz, Pz-Oz), horizontal EOG, chin EMG, event marker. 153 SC files (1987–1991 study of age effects, healthy subjects aged 25–101) plus 44 ST files (1994 temazepam study, 22 subjects). EEG at 100 Hz. Hypnograms W, R, 1, 2, 3, 4, M, ? (R&K).
- 8.1 GB; Open Data Commons Attribution v1.0; open download (no login).
- **Literature protocols:** Sleep-EDF-20 (39 recordings from 20 subjects, Fpz-Cz, 30-s epochs, 20-fold subject-wise CV, from the DeepSleepNet line) and Sleep-EDF-78 (197 recordings from 78 subjects, 10-fold subject-wise CV).
- https://physionet.org/content/sleep-edfx/

#### C39 — TUH EEG corpus (Temple University): TUEG, TUAB, TUEV
- Access: free, but you have to email a signed request form to help@nedcdata.org and then download by rsync with the credentials you get back. Licence terms are on the form (not read).
- TUEG: 26,846 clinical EEG recordings (2002–2017), the usual large unlabelled pool for EEG pretraining.
- TUAB v3.0.1: normal/abnormal; 2,993 EDF files, 60 GB. LaBraM describes it as 23 channels at 256 Hz.
- TUEV: six classes (SPSW, GPED, PLED, EYEM, ARTF, BCKG); train and eval partitions.
- **Protocol:** TUAB and TUEV are the standard downstream pair for EEG foundation models (BIOT, LaBraM, CBraMod, ...). LaBraM states it follows BIOT's TUAB/TUEV splitting exactly (BIOT's split details not checked).
- https://isip.piconepress.com/projects/nedc/html/tuh_eeg

#### C40 — CHB-MIT Scalp EEG (PhysioNet)
- 23 cases from 22 paediatric subjects with intractable seizures; 256 Hz, 16-bit; most files have 23 EEG signals and are one hour long. 198 seizures; 42.6 GB; ODC-By 1.0; open.
- Continuous recordings with rare seizures, so very imbalanced. No single standard split found.
- https://physionet.org/content/chbmit/

#### C41 — BCI Competition IV dataset 2a (MOABB: BNCI2014_001)
- 9 subjects, 22 EEG channels at 250 Hz, 4 motor-imagery classes (left hand, right hand, feet, tongue). Two sessions on different days, 288 trials each.
- **Standard split:** the competition's split, with the first session as labelled calibration data and the second as evaluation (its true labels have since been released). Loadable through MOABB.
- Why: shift between sessions (same subject, another day); very few labels.
- Licence not confirmed. https://bbci.de/competition/iv/

#### C42 — PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)
- 109 volunteers, over 1,500 one- and two-minute recordings, 64 channels at 160 Hz (BCI2000). 14 runs per subject: eyes-open/closed baselines and executed/imagined fist or feet movements.
- No single standard split found; licence not checked. Many subjects, so it suits cross-subject splits.
- https://physionet.org/content/eegmmidb/

#### C43 — Sleep Heart Health Study (SHHS, NSRR) — low priority
- 5,804 adults aged 40+; about 350 GB of EDF. Requires an approved Data Access and Use Agreement.
- Sleep-staging papers use a 5-fold subject-wise protocol.
- https://sleepdata.org/datasets/shhs

### ECG

#### C44 — PTB-XL (PhysioNet, v1.0.3)
- 21,799 clinical 12-lead ECGs (10 s) from 18,869 patients; 500 Hz plus a 100 Hz downsampled copy; 71 SCP-ECG statements (multi-label: diagnostic, form, rhythm). About 3 GB; CC BY 4.0.
- **Standard split by the authors:** `strat_fold` 1–10 (patient-wise, stratified); folds 1–8 train, 9 validation, 10 test. Folds 9–10 had at least one human evaluation of their labels.
- Why: multichannel data with a clean, official patient-wise split, and it is large.
- https://physionet.org/content/ptb-xl/

#### C45 — MIT-BIH Arrhythmia (PhysioNet)
- 48 two-channel records of about 30 minutes each. Beats are grouped into the AAMI EC57 classes (N, S, V, F, Q).
- **Standard split:** de Chazal et al. (2004) inter-patient split: DS1 (22 records) for training and DS2 (22 records) for testing, so no patient is on both sides. Sampling rate (360 Hz) not rechecked.
- https://physionet.org/content/mitdb/

#### C46 — PhysioNet/CinC Challenge 2017 (AF classification)
- 8,528 single-lead AliveCor ECGs, 9–61 s, 300 Hz; classes Normal, AF, Other, Noisy. The 3,658-record hidden test set was never released, so papers build their own splits.
- https://archive.physionet.org/pn3/challenge/2017

### Human activity / wearables

#### C47 — UCI HAR (Human Activity Recognition Using Smartphones)
- 30 volunteers wearing a waist-mounted smartphone; 3-axis accelerometer and gyroscope at 50 Hz; 6 activities. Windows of 128 samples.
- **Official subject-wise split:** 21 volunteers train / 9 test, giving 7,352 / 2,947 windows. 58 MB.
- Used by TS-TCC and AdaTime. https://archive.ics.uci.edu/dataset/240

#### C48 — HHAR (Heterogeneity Activity Recognition, UCI)
- 9 users; 8 smartphones and 4 smartwatches of several models; accelerometer and gyroscope at each device's maximum rate (about 50–200 Hz); 6 activities (biking, sitting, standing, walking, stairs up/down).
- No official split; AdaTime's scenarios (C37) are the common protocol.
- Why: shift between devices plays the role of the sensor/rig nuisance.
- https://archive.ics.uci.edu/dataset/344

#### C49 — CAPTURE-24 (Oxford, Scientific Data 2024)
- Wrist-worn Axivity AX3 data from 151 participants, about 24 h each, free-living; 3,883 h in total, 2,562 h annotated. CC BY 4.0.
- Why: lots of unlabelled and labelled data from many subjects (scale for pretraining). No standard split checked.
- https://doi.org/10.5287/bodleian:NGx0JOMP5

### Audio (high sampling rate)

#### C50 — Google Speech Commands v2
- 105,829 one-second clips, 35 words, 16 kHz mono WAV. Fixed split through `validation_list.txt` / `testing_list.txt`. CC BY 4.0.
- Why: a high-rate 1-D signal with a standard split, useful for the open problem of high sampling rates.
- https://arxiv.org/abs/1804.03209

#### C51 — ESC-50
- 2,000 environmental sound clips (5 s, 44.1 kHz, mono), 50 classes. 5 predefined folds; clips from the same source recording are kept in one fold. CC BY-NC (ESC-10 subset: CC BY).
- https://github.com/karolpiczak/ESC-50

### Protocol sources worth watching (EEG)
- EEG-FM-Bench (arXiv 2508.17742), AdaBrain-Bench (arXiv 2507.09882), NeuroAtlas (arXiv 2605.14698; 42 datasets), OmniEEG-Bench (arXiv 2606.00815; 54 datasets): recent attempts to fix unified EEG protocols. Not read beyond abstracts.

### Added later on 2026-11-18: resting-state EEG and more non-EEG sets

#### C52 — Resting-state EEG: Alzheimer's, frontotemporal dementia and healthy controls (Miltiadous et al. 2023; OpenNeuro ds004504)
- 88 subjects (36 AD, 23 FTD, 29 healthy controls), resting state with eyes closed; 19 scalp electrodes (10–20 system), 500 Hz. Recordings average roughly 12–14 min per group. Raw and preprocessed EEG in BIDS; MMSE score per subject.
- Licence CC0. On OpenNeuro (whether download needs a login was not checked).
- **No official split.** The authors use leave-one-subject-out (LOSO) on 4-s epochs with 50% overlap, averaging over subjects. LOSO is the common protocol, but it means 88 folds per method (compute).
- https://openneuro.org/datasets/ds004504 ; paper https://www.mdpi.com/2306-5729/8/6/95

#### C53 — Other resting-state EEG sets (no split; low priority under the "official split" criterion)
- Test-retest resting and cognitive-state EEG (Wang et al., Scientific Data 2022): 60 participants, 3 sessions (two within 90 min, one a month later); eyes open/closed plus memory, music and subtraction states. https://doi.org/10.1038/s41597-022-01607-9
- Dortmund Vital Study (Getzmann et al., Scientific Data 2024): 64-channel resting EEG, 608 participants aged 20–70, eyes open/closed before and after 2 h of tasks; 5-year follow-up for 208. BIDS on OpenNeuro. https://doi.org/10.1038/s41597-024-03797-w

#### C54 — Monash/UEA/UCR Time Series Extrinsic Regression (TSER) archive
- 19 regression datasets (energy, environment, health incl. PPG heart rate and BIDMC respiration/HR/SpO2, sentiment, forecasting). **Fixed train/test files ship with the archive**: random 70/30 for most; time-based for some (e.g. HouseholdPowerConsumption, BeijingPM); subject-based for PPGDalia (subjects 1–10 train); the original competition split for IEEEPPG.
- `.ts` format, loadable with sktime/aeon. Licence not stated in the paper (depends on each source).
- Why: the only standard archive for whole-series **regression**, the task type of κ estimation.
- http://timeseriesregression.org/ ; https://arxiv.org/abs/2006.10996

#### C55 — DCASE 2021 Task 2 (MIMII DUE + ToyADMOS2): machine sounds under domain shift
- Seven machine types: fan, gearbox, pump, slide rail, valve (MIMII DUE, real machines) plus ToyCar and ToyTrain (ToyADMOS2). First channel only (single fixed microphone). Anomalies come from intentionally damaged machines.
- **Official split:** development set (normal-only train; test with normal and anomalous), additional training set and evaluation set (about 200 clips per machine type, section and domain). Source and target domains are defined per section. Ground truth for the evaluation set was released after the challenge.
- CC BY-NC-SA 4.0 (stated on the evaluation-set record). Zenodo.
- Why: non-bearing machinery (including a gearbox), operating-condition domain shift, and a fixed protocol. Complements C14/C29 (bearing-only).
- Dev https://zenodo.org/record/4562016 ; additional training https://zenodo.org/record/4660992 ; eval https://zenodo.org/record/4884786 ; task paper https://arxiv.org/abs/2106.04492

#### Note on C35 (TF-C sets), checked on the TF-C README 2026-11-18
- Each dataset comes as three `.pt` files (fixed partitions). Fine-tuning splits (train/val/test): Epilepsy 60/20/11,420; Gesture 320/120/120; EMG 122/41/41; FD-B 60/21/13,559. Pretraining sets: SleepEEG 371,055 samples, HAR 10,299, ECG 43,673, FD-A 8,184.
- README rule: train:val about 3:1, classes balanced where possible, the rest is test. Samples from one recording stay in one partition where possible, except Epilepsy (a re-split Epilepsy version was added later).
- Licences per README: SleepEEG, ECG, EMG ODC-By 1.0; Epilepsy CC BY 4.0; FD-A/B CC BY-NC 4.0; HAR no commercial use; Gesture none named. Code MIT.

---

## Shortlist by access and split (Claude's ranking, 2026-11-18)

Criteria (the user's, 2026-11-18): publicly and easily available (no form, no data-use agreement), and an official split or one common split, so that few decisions are needed.

**Tier 1: open download and a fixed official split (no split decision)**

| Entry | Dataset | Domain | Split |
|---|---|---|---|
| C36 | UCR / UEA (+ Multiverse) | mixed | fixed train/test files |
| C54 | TSER archive | mixed, **regression** | fixed train/test files |
| C35 | TF-C sets (Epilepsy, Gesture, EMG; SleepEEG/HAR/ECG for pretraining) | EEG, accelerometer, EMG, ECG | fixed `.pt` partitions |
| C44 | PTB-XL | 12-lead ECG | `strat_fold` 1–8 / 9 / 10 (patient-wise) |
| C47 | UCI HAR | accelerometer, gyroscope | 21 / 9 volunteers |
| C41 | BCI Competition IV 2a | EEG | session 1 train / session 2 test |
| C45 | MIT-BIH | ECG | de Chazal DS1 / DS2 (common; not official) |
| C37 | AdaTime (UCIHAR, WISDM, HHAR, SSC) | accelerometer, EEG | fixed source→target scenarios |
| C50 | Speech Commands v2 | audio, 16 kHz | validation/testing lists |
| C55 | DCASE 2021 Task 2 | machine sound | dev / eval |

**Tier 2: open, but the common protocol is cross-validation (no decision, more runs)**

| Entry | Dataset | Protocol |
|---|---|---|
| C38 | Sleep-EDF-20 | 20-fold subject-wise, i.e. leave-one-subject-out over 20 subjects |
| C51 | ESC-50 | 5 predefined folds (non-commercial licence) |
| C52 | ds004504 resting-state EEG | LOSO, 88 folds |

**Drop under these criteria:** TUAB/TUEV (C39, signed form), SHHS (C43, data-use agreement), CHB-MIT (C40), EEGMMIDB (C42), CAPTURE-24 (C49), PhysioNet 2017 (C46; hidden test set never released), other resting-state sets (C53) (no standard split). HHAR (C48) only through AdaTime's scenarios.

#### C56 — Resting-state EEG for sleep deprivation (Xiang et al., Scientific Data 2024; OpenNeuro ds004902, v1.0.8) — added 2026-11-18
- 71 participants, each recorded in two sessions: normal sleep (NS, session 1) and sleep deprivation (SD, session 2). The session labels are not the time order, which was counterbalanced and is given in the metadata. Resting state, eyes open and partly eyes closed.
- 61 electrodes (Brain Products), 500 Hz. Sleepiness and mood questionnaires per participant. Collected March 2019 – December 2020. CC0 (per the EEGDash listing: 218 recordings).
- **No official or common split found.** The natural task (NS vs SD, the same people in both classes) needs a subject-wise split chosen by us. Under the 2026-11-18 criteria it belongs with the "drop" group or tier 2. Not downloaded.
- https://openneuro.org/datasets/ds004902/versions/1.0.8 ; paper https://doi.org/10.1038/s41597-024-03268-2

### Downloaded 2026-11-18 (tier 1; downloaded only, not adopted, no decisions made)
Raw files in `data/raw/` (unpacked by `scripts/unpack_raw.py`, which deletes each archive after checking every file size):
- `ucr_archive/` (Univariate2018_ts), `uea_archive/` (Multivariate2018_ts), `tser/<dataset>/` (19 datasets): UEA-hosted aeon-toolkit mirror, timeseriesclassification.com/aeon-toolkit.
- `tfc_sleepeeg/`, `tfc_epilepsy/` (figshare version 2, the re-split), `tfc_har/`, `tfc_gesture/`, `tfc_ecg/`, `tfc_emg/`: TF-C figshare items (`train.pt`/`val.pt`/`test.pt`). FD-A/FD-B were already in `fd_a/`, `fd_b/`.
- `ptbxl/` (PhysioNet v1.0.3), `mitbih/` (PhysioNet v1.0.0), `uci_har/` (UCI repository id 240).
- `bci_iv_2a/` (A01T–A09E.mat from BNCI Horizon 2020, data set 001-2014; CC BY-ND 4.0).
- `adatime_har/`, `adatime_hhar/`, `adatime_wisdm/`, `adatime_ssc/` (NTU DR-NTU dataverse, CC BY-NC 4.0; AdaTime's FD set not downloaded because of the Paderborn overlap).
- `speech_commands_v2/` (download.tensorflow.org, v0.02).
- `dcase2021_task2/dev_data/`, `eval_data/` and `ground_truth_data/` (Zenodo 4562016, 4660992, 4884786, 5257674; CC BY-NC-SA 4.0).
