# Dataset candidates

Running log of datasets that could be used to test the methods, beyond the ones in use.
Claude's suggestions are listed here; they are not decisions. When a dataset is adopted,
record it in `decisions/` (D-number) and mark the entry here as "Adopted as Dn";
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

#### C20 — Paderborn run-to-failure under time-varying load and speed (KAt / Chair of Dynamics and Mechatronics, 2024)
- 17 run-to-failure experiments (B01–B17) on ball bearings, no initiated defect (natural damage); time-varying load and speed. 152 GB as ZIP archives plus a PDF description. CC BY 4.0.
- Sensors, sampling rates and profiles not stated on the record page (in the PDF description, not read).
- Why: natural degradation under non-stationary operation; split by bearing possible. Different recordings from the Paderborn KAt fault data (D47), same university.
- https://zenodo.org/records/10868257 ; paper: Javanmardi et al., PHM Society European Conference 2024, doi 10.36001/phme.2024.v8i1.4101

#### C21 — HUST bearing (Huazhong University of Science and Technology; not the Hanoi HUST of C1)
- ER-16K bearing on a SpectraQuest simulator; 9 health states (normal; inner, outer, ball, inner+outer, each medium/severe); 11 speed conditions (10 constant, 20–80 Hz, plus 0-40-0 Hz time-varying); one accelerometer, 25.6 kHz, 262,144 points per file; 99 Excel files.
- Licence not stated ("public"). Download via Google Drive / Quark links on the GitHub page.
- Why: many speed domains; used in a domain-generalization benchmark (Zhao, Zio & Shen 2024, RESS, doi 10.1016/j.ress.2024.109964), which may give a standard cross-domain split.
- Naming clash with C1: needs a distinct name in the harness if adopted.
- https://github.com/CHAOZHAO-1/HUSTbearing-dataset

#### C22 — MCC5-THU gearbox under variable working conditions (Tsinghua, 2024)
- Two-stage parallel gearbox; faults on a gear and its adjacent support bearings: pitting, wear, tooth break, tooth crack, inner race, outer race, compound faults, several severities.
- Two triaxial accelerometers (motor drive end, intermediate-shaft bearing seat), 12.8 kHz; speed and torque channels. 12 working conditions with speed and/or load varying over 60 s runs; 240 CSV files. MIT licence (GitHub).
- Use Mendeley version 2 (version 1 has reversed torque signs; torque needs a scaling factor of about 6, per the repo).
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
- SpectraQuest rig, two bearings; 8 accelerometer channels (radial, axial, tangential), tachometer, microphone; 50 kHz, 5 s per sequence; 1,951 sequences. Classes: normal (49), imbalance (333), underhang bearing faults (558), overhang bearing faults (513), horizontal (197) and vertical (301) misalignment; normal at 737–3,686 RPM.
- Figures as reported in arXiv 2301.02243; the official page blocks automated access (not checked there). Licence not checked. A gated Hugging Face copy exists (AI4Manufacturing/MAFAULDA-rotor-perception).
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
- Helicopter accelerometers; 1,024 Hz; 1-min sequences (61,440 samples). Train: 1,677 normal sequences; test: 594 (297 normal, 297 anomalous); labels per flight. Normal-only training.
- Not bearings specifically. Figures from Garcia et al. (arXiv 2005.07031); licence not stated on the record ("open access", rights holder Airbus SAS).
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
