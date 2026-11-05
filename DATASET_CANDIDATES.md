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
