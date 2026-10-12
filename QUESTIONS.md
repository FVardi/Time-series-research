# Open questions (for supervisor discussion)

Questions that need a decision but are better discussed first. Each entry: the
question, why it matters, the options with their trade-offs, and Claude's
suggestion (clearly marked, not a decision). When a question is settled, record the
decision in `decisions/` (D-number) and mark the entry here as "Settled by Dn" —
entries are not deleted.

---

## Q1 (2026-10-09) — Herning: hold out a temperature block, or split mixed sweeps?
**Context.** In the Herning run the lubricant does not change, so κ is a fixed
function of rpm and temperature (D55). The run has 13 temperature targets, 40–100 °C
in 5 °C steps, about 305 sweeps each (plus 5 cool-down sweeps at target 25).
D56 says "temperature hold-out", details open.

**Options.**
- *Mixed sweeps, grouped by acquisition hold* (as in the MSSP `grouping.py`, with
  "twin" holds merged). Every test sweep has training sweeps at almost the same rpm
  and temperature, so a model can predict κ well by recognising the operating point.
  Measures interpolation within known conditions (optimistic).
- *Hold out a temperature block.* The model meets a temperature it has never seen.
  Stricter; matches the use case (monitoring under unseen conditions) and the
  research plan (Step 1: "held-out temperature block").
  - Which block? A middle block tests interpolation; the hottest or coldest block
    tests extrapolation, and its κ values lie partly outside the training range
    (linear probes extrapolate poorly).
  - One block or rotate (leave each block out in turn)? Rotation is more robust,
    but every SSL method must then be pretrained once per block.
  - Validation block, or cross-validation inside the training blocks?

**Claude's suggestion.** Temperature hold-out as the main split (a middle block),
with the hold-grouped split reported alongside so the gap is visible.

**Status.** Open — user to discuss with supervisor.

---

## Q2 (2026-10-09) — Comparing whole-sweep features with window-based SSL representations
**Context.** The feature baseline gives one vector per sweep (14 statistics over the
whole 0.2 s sweep, D54). The SSL methods cannot take a whole sweep (500,000 samples
at 2.5 MHz; even after downsampling, D51) and give one vector per window, with many
windows per sweep. κ is one value per sweep.

**Why it matters.** If probes are trained on window vectors, they get many more,
noisier examples with κ repeated per window — a different task from predicting from
one vector per sweep, so the comparison would not be fair.

**Options.**
- (a) Average the window vectors of a sweep into one vector per sweep.
- (b) Take the element-wise maximum over windows (as TS2Vec pools over time).
- (c) Compute the features per window instead of per sweep (reverses D54 part 2).

**Claude's suggestion.** (a): the comparison is then per sweep, like the features.

**Status.** Open — user to discuss with supervisor.

---

## Q3 (2026-10-09) — US above 40 kHz and AE above 560 kHz: bearing signal or electrical interference?
**Context.** Band check D52 (`runs/20261009_085307_band_check_herning`, figure `band_check.png`).
US bands from about 40 to 200 kHz are only 0.6–2.6 dB above the non-rotating sweeps,
yet their power rises with rpm *within* each temperature block (median Spearman
0.6–0.85) and correlates with κ (about 0.8). AE bands above about 560 kHz show no
excess power but *fall* with rpm within blocks (about −0.7). Correlation within a
block rules out slow drift over the run, but not electrical interference that also
scales with speed — for example from the frequency converter (VFD), whose switching
noise changes with speed and load.

**Why it matters.** It decides the downsampling cut-off for US (D51) and whether
these bands should be in the features: keeping interference teaches models to read
the drive's electrical state instead of the bearing.

**Ways to find out.** Compare with a recording where the VFD runs but the bearing
is decoupled or not loaded; check whether the high-band content contains the VFD
switching frequency and its harmonics; ask the rig team about shielding/grounding.

**Status.** Partly answered by the user (2026-10-10): AE above 560 kHz is noise (D61, D62). US above 40 kHz may carry information and is kept (US bands up to 178 kHz, D60; US downsampled to 500 kHz, D62); whether it is bearing signal or interference is still open — to discuss with supervisor / rig team.
