# Method ideas (own SSL method for lubrication condition monitoring)

Running log of ideas for the user's own SSL method. Each entry: the idea, where it
comes from, what it assumes about the data, why it might suit lubrication
monitoring, risks, and open questions. Claude's suggestions are clearly marked and
are not decisions. When an idea is adopted, record it in `decisions/` (D-number)
and mark the entry here as "Adopted as Dn"; entries are not deleted.

---

## M1 (2026-10-30) — Partitioned representation: operating point / lubrication state / nuisance

**Motivation.** On the current Herning run κ is a fixed function of rpm and
temperature (D55, see also Q1). An encoder can predict κ well just by recognising
the operating point, without sensing anything about the lubricant film. Departures
from nominal lubrication (starvation, contamination, aged or wrong oil, degradation)
are what break this coupling. Goal (user): both estimate κ and detect departures
from nominal lubrication. Herning will later vary lubrication variables and include
degradation; load may be varied later (not now).

**Idea (user's current direction).** Split the representation into blocks:
- `z_op` — operating point (rpm, temperature; later load). Supervised during
  training by small heads that predict the measured metadata. Metadata only needed
  at training time; at test time `z_op` is still produced from the signal, so it can
  be kept as a κ estimator or dropped for state detection.
- `z_state` — lubrication state / departure from nominal: what the operating point
  does not explain.
- `z_id` — nuisance: bearing, rig, sensor mounting.

Mechanism the user prefers: adversarial. `z_op` is rewarded for allowing a small
head to predict the operating point; the other blocks are punished for it (e.g.
gradient reversal). Possibly also orthogonality between `z_state` and `z_id`.

**Expected behaviour on current Herning.** `z_state` should carry ~no κ information
and `z_op` nearly all of it. This is a sanity check of the split, not a failure.

**Open questions.**
1. How are `z_state` and `z_id` directed differently? Orthogonality alone is
   symmetric (swapping/rotating the blocks satisfies it equally), so each block
   needs its own asymmetric signal. Options:
   - *Labels:* a head predicts installation identity from `z_id`; `z_state` is
     punished for it. Risk: run ID is confounded with lubrication state if each
     run has its own lubricant, so the label must vary independently of
     lubrication (needs a crossed design: same bearing under several lubrication
     states, different bearings under the same state).
   - *Grouping / time structure:* `z_id` constant within an installation, varies
     between; `z_state` varies slowly within a run beyond what `z_op` explains.
     Cf. FHVAE (sequence- vs segment-level factors) and ML-VAE (grouped
     observations).
   - Both combined.
2. Base SSL objective (JEPA, contrastive, reconstruction). Needed so `z_state`
   contains anything at all — the adversarial/orthogonality terms alone are
   trivially satisfied by `z_state = 0`. Each block probably needs an anti-collapse
   term (e.g. VICReg-style variance). Ties into the signal-character and
   representation discussion (bursty AE, rpm periodicity, window-feature sequences).
3. Are rpm/temperature available at deployment? Not known yet; the design above
   only needs them during training.

**Risks.**
- Adversarial removal leaks: information hidden from the training adversary can be
  recovered by a stronger probe (Elazar & Goldberg 2018). Evaluation probes must be
  stronger than the adversary.
- Min-max training can oscillate, especially with continuous targets (rpm, T).
  Non-adversarial fallback: an independence penalty such as HSIC between `z_state`
  and the metadata.
- Removal deletes everything correlated with the removed variable in the training
  data, so the confound structure of the data decides what can be separated.

**Evaluation idea.** Probe every block for every factor (rpm, T, κ, installation
ID, later lubrication state) → factor × block matrix; ideally diagonal. Reuses the
harness probes. Related metric: DCI (Eastwood & Williams 2018) — not yet checked.

**References.**
- Locatello et al. 2019, Challenging common assumptions in the unsupervised
  learning of disentangled representations (ICML) — https://arxiv.org/abs/1811.12359
- von Kügelgen et al. 2021, SSL with data augmentations provably isolates content
  from style (NeurIPS) — https://arxiv.org/abs/2106.04619
- Xiao et al. 2021, What should not be contrastive in contrastive learning (ICLR)
  — https://arxiv.org/abs/2008.05659
- Tsai et al. 2022, Conditional contrastive learning with kernel (ICLR)
  — https://arxiv.org/abs/2202.05458
- Hsu, Zhang & Glass 2017, FHVAE (NeurIPS) — https://arxiv.org/abs/1709.07902
- Bouchacourt et al. 2018, ML-VAE (AAAI) — https://arxiv.org/abs/1705.08841
- Elazar & Goldberg 2018, Adversarial removal of demographic attributes from text
  data (EMNLP) — https://arxiv.org/abs/1808.06640
- Bearing-domain disentanglement (supervised fault diagnosis, read as abstracts only):
  Du et al. 2025 (DVAE, global/local split, CWRU); Wang et al. 2026 (BYOL + HSIC,
  Meas. Sci. Technol.); Chang et al. 2024 (CDHM, IEEE TNNLS); Vásquez et al. 2026
  (XCDN, gradient reversal to split fault vs RPM, e-J. NDT); Zhao et al. 2026
  (orthogonality fault vs condition, Mathematics).
- Possible gap (one search only, needs a proper sweep): label-free SSL with a
  metadata-supervised split, for lubrication monitoring with AE/ultrasound.

**Pilot proposal (2026-10-30, not yet approved).** Test the removal mechanism on a
dataset where health and operating condition are crossed, before Herning varies
lubrication. Backbone: TS2Vec (user's choice). Tests the mechanism, not lubrication
(CWRU is vibration, not AE).
- *Dataset:* CWRU (converted; every fault class recorded at loads 0-3 hp). Load
  changes speed only slightly (about 1730-1797 rpm), so the condition effect may be
  weak. Paderborn has larger condition changes but is not converted.
- *Step 0 (pre-check):* probe the existing TS2Vec CWRU representations
  (`runs/20261029_094121_ts2vec_cwru`) for `load_hp`. If load is not recoverable,
  there is nothing to remove and CWRU cannot test the mechanism.
- *Method:* new named method (D19), e.g. `ts2vec_split`, reusing the official
  TSEncoder, cropping and hierarchical contrastive loss in an own training loop.
  The 320-dim representation is split into `z_op` (k dims) and `z_state` (rest);
  TS2Vec loss on all dims; heads on the max-pooled crop representations.
- *Variants:* (0) plain TS2Vec (existing run); (A) split + load head on `z_op`, no
  removal; (B) A + adversary on `z_state` via gradient reversal; (C) A + HSIC
  between `z_state` and load.
- *Evaluation:* factor x block matrix: probes for fault class and load on `z_op`,
  `z_state` and the full representation; probes at least as strong as the adversary;
  label fractions and seeds as for the CWRU benchmark.
- *Open choices (to decide):* condition label; block size k; adversary form, update
  scheme and weight schedule; loss weights; HSIC kernel; batch size; probe types.

**Status.** Under discussion; pilot proposed.
