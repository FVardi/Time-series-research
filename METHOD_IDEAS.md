# Method ideas (own SSL method for lubrication condition monitoring)

Running log of ideas for the user's own SSL method. Each entry: the idea, where it
comes from, what it assumes about the data, why it might suit lubrication
monitoring, risks, and open questions. Claude's suggestions are clearly marked and
are not decisions. When an idea is adopted, record it in `decisions/` (D-number)
and mark the entry here as "Adopted as Dn"; entries are not deleted.

Working approach (user, 2026-11-03): this file is a bank of ideas. The own method starts
simple; components are introduced one at a time and ablated, so every added piece has
to show a measurable gain (accuracy / kappa, generalisation, label efficiency, speed).

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

**When should (part of) the representation be invariant? (discussion 2026-11-03).**
M1 does not make the whole representation invariant: `z_op` keeps the operating point,
only `z_state` is pushed to be free of it. The question is whether such an invariant
block helps, and the synthetic designs (D196) give different expected answers:
- *crossed:* state independent of the operating point. Invariance should not hurt, and
  can only help when the test conditions differ from training (held-out operating
  points); on random splits a benefit is not expected.
- *confounded:* state = g(operating point). Removing the operating point from `z_state`
  removes the state too, by construction; the state is then only available via `z_op`.
  Shows the cost of invariance.
- *departures:* the case M1 is built for. Diagnostic (Claude's suggestion): probe trained
  on nominal runs, tested on departure runs; does it predict the actual state or the
  nominal one? A shortcut representation predicts the nominal state.
Prediction (Claude's, to test): the removal only keeps state information in `z_state`
if the *pretraining* data contains state variation at fixed operating points (departure
runs, unlabelled is enough). Pretrained on purely confounded data, the adversary also
removes the bursts, because they predict the operating point there. For Herning this
would mean the split pays off only once lubrication-varied runs exist.

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

**Pilot (2026-10-30, approved as D182-D184).** Test the removal mechanism on a
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
- *Settled:* D182-D185 (both CWRU splits, D185).

**Step 0 result (2026-10-30).** Logreg probe for `load_hp` (4 classes), trained on the
D128 bearing-split training windows (7/14 mil), tested on unseen 21 mil bearings; all
labels; `studies/split_pilot/results/load_probe.csv`.
- TS2Vec (`runs/20261029_094121_ts2vec_cwru`): accuracy 0.41 / 0.40 / 0.42 (seeds 0-2).
- |FFT| (`runs/20261029_092116_fft_cwru`): 0.965 (identical over seeds; features do not
  depend on the seed). Floor: 0.25.
- Reading: load is almost fully recoverable from the spectrum, so CWRU can test the
  mechanism. Plain TS2Vec keeps only part of it (linear probe; an MLP probe may find
  more). Caveat: D128 bearing split, not the D163 main split.

**Synthetic pilot results (2026-11-04; seed 0; logreg, all labels; `studies/split_pilot/results/block_probes/`).**
- The heads work: on the random split z_op holds speed (1.00) and temperature (0.93) in 32 dims, and the
  state sits in z_state (0.80-0.82) rather than z_op (0.47-0.49).
- Removal fails: z_state still predicts speed at 0.98-1.00 in A, B1 and C1. On departures the state probe
  on z_state predicts the nominal state for 57-72 % of the departure windows (plain TS2Vec, full: 60 %).
- No variant carries the state over to an unseen speed (all about chance, 0.23-0.38 vs 0.33).
- B1: the adversary's loss stays at chance (ln 4 + ln 3) for the whole run. The live encoder's z_state
  leaks even more than the averaged copy's (temperature 0.97 vs 0.65; nominal-state predictions 0.94 vs
  0.59), so the weight averaging is not what keeps the information.
- C: the HSIC term stays at about 0.01 at every weight. At weights 100 and 1000 it does not go down;
  instead the TS2Vec loss stays high (about 4.2 vs 1.0) and the op heads stay near chance, the speed in
  z_state drops only partly (0.60-0.85) and the state is lost (random split, full: 0.49 vs 0.83).

**Status: parked (user, 2026-11-04; D208).** Step 0 and the synthetic runs are done; removal does not
work yet. To pick it up again (Claude's suggestions, none decided):
1. Check that the adversary can learn at all: B with removal weight 0 (adversary trained, nothing
   reversed). Loss well below chance -> the min-max game is the problem; still at chance -> what the
   adversary sees (short random crops vs full windows) or its setup.
2. A stronger adversary: several adversary steps per encoder step, or the adversary on full windows.
3. A removal measure the encoder can reduce, e.g. a linear cross-covariance penalty (plain HSIC with the
   median-heuristic bandwidth did not go down at any weight; that the bandwidth is the cause is unchecked).
4. Weaker periodic lines in the generator: speed dominates the toy signal, possibly more than in real data.
Not run: the CWRU pilot configs (`pilot_*_cwru*.yaml`; batch 64 on 2,048-sample windows may exceed 24 GB),
B0.1/B10/C0.1/C10, the MLP probe (pilot item C).

---

## M2 (2026-11-03) — Discrete targets next to latent regression (from TimeMAE's MCC)

**Status: parked (user).** Not an initial direction. Revisit once the base objective
and the block split (M1) are established. Logged so it is known this has been looked at.

**Where it comes from.** TimeMAE (Cheng et al., arXiv 2303.00320, WSDM'26) scores the
masked-patch predictions with two losses: MRR, an MSE to the representations of an EMA
target encoder (BYOL / data2vec / JEPA family), and MCC, a cross-entropy to discrete
codeword labels of the masked patches produced by a tokenizer. The paper motivates MCC
by analogy to masked word prediction and product quantisation; the combination looks
ported from CAE (Chen et al. 2022), where the discrete targets come from a pretrained
DALL-E tokenizer. TimeMAE learns its tokenizer inside the training loop instead.

**What the official code does** (`model/TimeMAE.py`, `loss.py`, `args.py`; read
2026-10-28 and 2026-11-02):
- Tokenizer = one linear layer d -> K (K = 192 by default) applied to the raw patch
  projection, before the positional embeddings are added, so labels carry no position.
- Labels = argmax of `F.gumbel_softmax(logits)`, i.e. a fresh sample from
  softmax(logits) each step, with no gradient through the labels. The paper's
  straight-through estimator is not in the code.
- The same layer is the prediction head for the regressor outputs (cross-entropy,
  label smoothing 0.2). No commitment loss and no codebook update towards the data,
  so the codebook only learns as a classifier.
- Loss weights: code defaults 5 x MRR + 1 x MCC; the paper fixes the MCC weight at 1
  and searches the MRR weight in 1-10.
- No MCC-only / MRR-only ablation found in the arXiv v1-3 or v4 text (PDF not checked).

**Why it might help (Claude's reasoning, not verified).**
1. Anchor against encoder collapse: the labels come from the raw projection, not the
   encoder, so a collapsed encoder cannot make MCC trivial. Only partial: the
   projection is trained too, and if it shrank both losses would become trivial.
2. Noise robustness: a label only changes when a patch crosses a decision boundary
   (cf. wav2vec 2.0 quantised targets). Possibly relevant for noisy AE / ultrasound.
3. Bounded, scale-free loss, unlike an MSE to moving EMA targets that can be
   dominated by a few high-variance dimensions.
4. The codebook may not need to be meaningful: BEST-RQ (Chiu et al. 2022, speech)
   uses a frozen random projection and random codebook as targets and works well.
   TimeMAE's tokenizer is close to a slowly drifting learned random-projection hash.

**Possible use in the own method (Claude's suggestion).** If the base objective ends
up JEPA-style (M1 open question 2), a discrete-target term could be an extra
anti-collapse / noise-robust term. With window-feature sequences as input, the
simplest variant is a frozen random-projection quantiser (BEST-RQ style) on the
features.

**Risks.**
- Codebook collapse: the same matrix produces and predicts the labels, and sharpening
  it lowers the loss on both sides (rich-get-richer). Codeword-usage perplexity
  would need logging.
- Fully collapsed solutions are still global minima of the TimeMAE loss; avoiding
  them relies on training dynamics (EMA, stop-gradient), not the objective.
- Discretising may throw away fine structure that matters for continuous targets
  (kappa, gradual lubrication degradation).

**Cheap test if revisited (not planned).** MRR only / MCC only / both with the TimeMAE
implementation on FordA, CWRU and Paderborn. The MRR-only variant is essentially
JEPA-style and would sit next to TS-JEPA and TF-JEPA. The TimeMAE baseline itself
still runs as published.

**References.**
- Cheng et al., TimeMAE (WSDM'26) — https://arxiv.org/abs/2303.00320 ;
  code https://github.com/ustc-time-series/TimeMAE
- Chen et al. 2022, Context Autoencoder (CAE) — https://arxiv.org/abs/2202.03026
- Bao et al. 2021, BEiT — https://arxiv.org/abs/2106.08254
- Chiu et al. 2022, BEST-RQ — https://arxiv.org/abs/2202.01855
- Baevski et al. 2020, wav2vec 2.0 — https://arxiv.org/abs/2006.11477

---

## M3 (2026-11-03) — Multimodal JEPA: within-modal plus cross-modal latent prediction (AE <-> US)

**Where it comes from.** A research-context note from another chat (user, 2026-10-31),
discussed here 2026-10-31 / 2026-11-03.

**Idea.** Separate, simple encoders for AE and US. Per modality, JEPA-style masked
latent prediction (predict a stop-gradient / EMA target encoder's representation of
masked time patches, no reconstruction of the input). Across modalities, bidirectional
prediction (AE context -> US target representation and US -> AE), first on global or
pooled targets, since the AE-US relation may be statistical rather than event by
event. Illustrative loss: L = L_within,AE + L_within,US + lambda L_cross.
rpm and temperature may help build training targets but are not model inputs at
deployment (same stance as M1).

**Input (user, 2026-11-03).** The compressed representation of D165, not raw samples
(raw 0.2 s sweeps at 2.5 MHz are far too large; the note's raw-CNN frontend is
overruled). The 48 D165 channels split by sensor: AE 4 bands x 4 features = 16
channels, UL 8 bands x 4 features = 32 channels; one sweep = 200 steps of 1 ms.

**Relation to M1.** Compatible. JEPA is a candidate for M1's base objective (M1 open
question 2); the M1 block split is an add-on to be justified by the pilot (D182). M1
keeps operating-point information in `z_op` instead of enforcing invariance, which
matches the note's warning against blanket invariance.

**Ablation sequence (from the note, adapted to the D165 features).**
1. AE-only JEPA. 2. US-only JEPA. 3. Both, pretrained independently, fused for the
probe. 4. Add bidirectional cross-modal prediction. 5. Compare with the harness
baselines (TS2Vec, TS-JEPA, TF-JEPA and the others on Herning, D173). Later and only if
justified: M4 items (time-frequency views, shared/private spaces, periodicity- or
burst-aware targets).

**Risk: cross-modal prediction may strengthen the operating-point shortcut.** (Noted
at the user's request.) Cross-modal prediction rewards whatever AE and US share, and
the operating point may be their strongest shared factor: both sensors see the rpm
periodicity and both are affected by temperature. So the cross-modal term could pull
the representation towards rpm and temperature rather than the film. Hypothesis, not
verified. Test: probe rpm, temperature and kappa on representations trained within-
modal only vs within + cross-modal (factor x block probes as D184). Possible remedy:
combine with M1, so `z_op` absorbs the shared operating-point information and the
cross-modal term in `z_state` has to rely on something else (also a hypothesis).
Related: on the current Herning run kappa = f(rpm, T), so leave-one-rpm-out or held-out
rpm-temperature combinations test generalisation across operating points, not film
sensing; only runs with changed lubrication at a fixed operating point separate the two.

**Splits (user, 2026-11-03, tentative).** Random splits as the default, as is standard
in the literature. To make a stronger generalisation point, add grouped / hold-out
splits: acquisition-grouped, leave-one-rpm-out, held-out rpm-temperature
combinations, the D56 temperature hold-out. Caveat (as D169): random splits put
near-duplicate neighbouring sweeps on both sides, so scores are optimistic.

**Open questions.** Masking scheme on 200-step feature sequences (patch length in
steps, mask ratio); cross-modal target form (pooled vs local-to-global); anti-collapse
(EMA alone, or a variance term); fusion of the two encoders for the probes; lambda.

**Window length check (from the note, numbers verified).** A 0.2 s sweep covers about
1.67 revolutions at 500 rpm and 10 at 3,000 rpm, which limits periodicity-based
targets inside one sweep at low speed.

**Status.** In the bank; not scheduled.

---

## M4 (2026-11-03) — Further ideas from the same note (deferred)

Each to be introduced only if an ablation shows a gain (working approach above).
- *Time-frequency consistency:* align waveform and spectral views (cf. the harness TF-C
  ablation, where the paper's full loss was unstable and positives worked: D122-D124).
- *Burst-specific targets:* burst density, duration, inter-burst intervals,
  amplitude / energy distributions. The D165 per-window kurtosis already carries part
  of this.
- *Periodicity-aware targets:* use approximate rpm, or bearing characteristic
  frequencies (BPFO, BPFI, BSF, FTF), to pick approximately phase-corresponding
  segments during pretraining, without giving rpm to the encoder. Do not force every
  component to be periodic; slip and speed changes make alignment approximate. First
  check whether a baseline representation already captures the periodic structure.
- *Contrastive objectives:* same-window or cross-modal positives. Negatives are hard:
  different acquisition groups need not be different lubrication states;
  VICReg / Barlow Twins-style regularisation avoids negatives.
- *Shared / private latent spaces across modalities:* common AE-US information vs
  sensor-specific. A different axis of factorisation from M1's blocks; lubrication
  information may sit in both.
- *Multi-scale architectures:* only if a simple encoder misses relevant time scales.
- *Augmentations (stance for the own method):* conservative. Amplitude or frequency
  scaling, time warping, filtering and aggressive cropping may remove lubrication
  information; use only augmentations justified by plausible measurement variation,
  and test their effect on kappa and generalisation. (The baselines keep their standard
  augmentations, D165 item 6.)
- *Principle:* do not read meaning into latent dimensions because of their names;
  show it with probes (D184).

---

## M5 (2026-11-03) — Synthetic test bed with known factors (for very quick tests)

**Why (user, 2026-11-03: wanted).** FordA has no operating-condition labels, so it cannot
test M1 or M3. A toy dataset where the factors are set by hand gives exact targets for
the factor x block probes (D184), is fast enough for the two-core VM, and separates
"the method fails" from "the dataset cannot show it". Results do not carry over to real
bearings by themselves: synthetic for the mechanism, then CWRU / Paderborn.

**Design (approved as D196).** Every formula and setting in one config;
values stored as generated (D5).
- *Factors per run:* speed (discrete levels) -> periodic lines at shaft harmonics and one
  characteristic order; temperature (discrete levels) -> broadband noise level /
  spectral tilt; lubrication state (classes or continuous) -> burst process (rate,
  amplitude, decay of short high-frequency bursts); "bearing" ID -> per-run gain and
  resonance (nuisance).
- *Designs:* crossed (every combination; mechanism test as CWRU), confounded (state =
  g(speed, temperature), as kappa on the current Herning run), confounded plus
  departures (some runs with state != g at a fixed operating point, as later Herning).
- *Two sensors (for M3):* AE-like (bursts strong) and US-like (other transfer function),
  sharing speed, temperature and state, each with a private nuisance.
- *Integration:* a canonical dataset `synthetic` written once by a generator script, so
  every method and config runs through the harness unchanged; labels and groups in
  `harness/labels.py`.

**Status.** Adopted as D196 (generator, configs, tests). Difficulty check on
synthetic_crossed (log |FFT| per 1,024-sample window, logreg, random 80/20 windows, not a
benchmark): speed 1.00, bearing 1.00, temperature 0.93, state 0.40 (chance 0.33), so the
default is the hard case: operating point and nuisance strong, state subtle.
