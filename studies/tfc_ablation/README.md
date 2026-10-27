# studies/tfc_ablation — TF-C study (D95)

Two parts, both built on our TF-C (`harness/methods/tfc.py`, D41, D42), which is
imported but never changed here (D11, D91):

1. **Examination of collapse** (D93): when and where TF-C's representations collapse
   during pretraining. — `collapse.py`
2. **Ablation by building blocks** (D96, D97): TF-C built up one component at a time. — `ablation.py`

## Collapse examination

    uv run python studies/tfc_ablation/collapse.py studies/tfc_ablation/configs/collapse_forda.yaml
    uv run python studies/tfc_ablation/collapse.py studies/tfc_ablation/configs/collapse_fd.yaml

Pretraining is exactly `tfc.pretrain` (same weights for the same seed; tested in
`tests/test_diagnostics.py`). Before training (epoch 0) and after every epoch, the
representations of a fixed set of pretraining windows are measured in evaluation mode
with `harness/diagnostics.py`. Output per seed (`runs/<date_time>_<name>/seed_<k>/`):

| file | contents |
|---|---|
| `pretrain_loss.csv` | per epoch: total loss and its parts L_T, L_F, L_C and S_TF (means over batches); the ablation also logs S_pair,1..3 (D101) |
| `collapse.csv` | per epoch and representation: n, d, finite, effective rank (RankMe), mean std, fraction of dead dimensions (std below 1 % of the largest std of that representation), mean norm, mean pairwise cosine, SIGReg statistic, alignment |
| `spectra.csv` | per epoch and representation: all covariance eigenvalues, k = 1 the largest |
| `pretrained.pt` | weights after the last epoch |

Representations: `h_t`, `h_f` (encoder outputs, flattened), `z_t`, `z_f` (projector
outputs) and `z` = [z_T; z_F] (what the probes use). In TF-C, L_T and L_F act on h,
only L_C on z.

Reading the measures: complete collapse = mean std near 0 and mean cosine near 1;
dimensional collapse = windows still differ (std > 0) but the effective rank is small
and most eigenvalues are near 0. If h collapses while z does not, the projector hides it.

`collapse_forda.yaml` is complete (D94). `collapse_fd.yaml` still has open placeholders (`null`: seeds and number of
diagnostic windows); the run refuses to start until they are set.

## Ablation by building blocks

    uv run python studies/tfc_ablation/ablation.py studies/tfc_ablation/configs/ablation_forda_A.yaml

| stage | branches | loss | probed on |
|---|---|---|---|
| A | time | L_T | h_T |
| B | frequency | L_F | h_F |
| C | both, uncoupled | L_T + L_F | h_T, h_F, h |
| D | both, aligned | λ(L_T + L_F) + (1 − λ) S_TF | h_T, h_F, h, z |
| E | full TF-C | λ(L_T + L_F) + (1 − λ) L_C | h_T, h_F, h, z |

Variants of D and E (D101), our constructions (not in the paper or the official code), to find
what in L_C collapses z. S_pair,1..3 are the three NT-Xent terms of L_C: (z_T, z̃_F), (z̃_T, z_F),
(z̃_T, z̃_F), with ~ the augmented view; L_C = 3 S_TF − ΣS_pair + 3δ, so E weights S_TF by 3(1 − λ).

| stage | loss | question |
|---|---|---|
| F | λ(L_T + L_F) + 3(1 − λ) S_TF | E without the pair terms: are they the cause (S_TF weighted as in E)? |
| G | λ(L_T + L_F) + (1 − λ) Σ max(0, S_TF − S_pair + δ) | hinged L_C: is the missing hinge the cause, or the pair terms themselves? |
| H | λ(L_T + L_F) + (1 − λ) (S_TF + ΣS_pair) / 4 | augmented cross-domain pairs as positives instead of negatives |

All three are probed and measured like D and E. In every stage with both branches (C–H),
`pretrain_loss.csv` also has `s_pair_1`, `s_pair_2`, `s_pair_3` (runs made before D101 lack them).

Per seed: pretraining with the collapse diagnostics after every epoch (same output files as
`collapse.py`), then frozen probes (D26: `svm_ts2vec` and `logreg`; label budgets 1/10/100 %)
trained on the training split and scored once on the test split: `probes.csv`, `summary.csv`.
Stage E is exactly `tfc.pretrain` with `loss.variant: paper`, and stage H exactly `tfc.pretrain` with
`loss.variant: positives`, the version used for the benchmarks (D122; both tested in `tests/test_ablation.py`). In C the projectors get no
gradient (only their BatchNorm running statistics move), so z is not evaluated there.

The probes need the TS2Vec submodule (`external/ts2vec`): `git submodule update --init`.

## Figures

Notebook `figures_forda.ipynb` (functions in `plots.py`, settings and run folders in
`configs/plots_forda.yaml` for stages A–E and `configs/plots_forda_variants.yaml` for D, E and
the variants F–H; add runs there once they have run):

    uv sync --group eda

then open the notebook (VS Code with the repo's `.venv`, or `uv run --group eda jupyter lab`)
and run all cells. Nothing is written to disk: the figures exist only in the notebook, which
is committed with outputs cleared (as in `eda/`, D68). Each section of the notebook says
what its figure shows and how to read it.

Figures: loss parts per epoch; effective rank, directions holding 90 % of the variance and
dead dimensions; cumulative variance at the last epoch and eigenvalue spectra per stage;
scale (norm, std); mean cosine, alignment and the two against each other; SIGReg; probe
accuracy against label budget with TS2Vec and T-Loss as reference; PCA of the diagnostic
windows at epoch 0 and the last epoch, coloured by class.

Only measured values are drawn. With several seeds a line is the mean over seeds and the
band runs from the smallest to the largest seed. A and B share one colour (they never share
a panel); every line carries its stage letter.
