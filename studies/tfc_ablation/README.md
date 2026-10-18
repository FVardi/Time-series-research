# studies/tfc_ablation — TF-C study (D72)

Two parts, both built on our TF-C (`harness/methods/tfc.py`, D41, D42), which is
imported but never changed here (D11, D71):

1. **Examination of collapse** (D73): when and where TF-C's representations collapse
   during pretraining. — `collapse.py`
2. **Ablation by building blocks**: TF-C built up one component at a time. — not started;
   stages open (D72).

## Collapse examination

    uv run python studies/tfc_ablation/collapse.py studies/tfc_ablation/configs/collapse_forda.yaml
    uv run python studies/tfc_ablation/collapse.py studies/tfc_ablation/configs/collapse_fd.yaml

Pretraining is exactly `tfc.pretrain` (same weights for the same seed; tested in
`tests/test_diagnostics.py`). Before training (epoch 0) and after every epoch, the
representations of a fixed set of pretraining windows are measured in evaluation mode
with `harness/diagnostics.py`. Output per seed (`runs/<date_time>_<name>/seed_<k>/`):

| file | contents |
|---|---|
| `pretrain_loss.csv` | per epoch: total loss and its parts L_T, L_F, L_C and S_TF (means over batches) |
| `collapse.csv` | per epoch and representation: n, d, finite, effective rank (RankMe), mean std, fraction of dead dimensions (std below 1 % of the largest std of that representation), mean norm, mean pairwise cosine, SIGReg statistic, alignment |
| `spectra.csv` | per epoch and representation: all covariance eigenvalues, k = 1 the largest |
| `pretrained.pt` | weights after the last epoch |

Representations: `h_t`, `h_f` (encoder outputs, flattened), `z_t`, `z_f` (projector
outputs) and `z` = [z_T; z_F] (what the probes use). In TF-C, L_T and L_F act on h,
only L_C on z.

Reading the measures: complete collapse = mean std near 0 and mean cosine near 1;
dimensional collapse = windows still differ (std > 0) but the effective rank is small
and most eigenvalues are near 0. If h collapses while z does not, the projector hides it.

`collapse_forda.yaml` is complete (D74). `collapse_fd.yaml` still has open placeholders (`null`: seeds and number of
diagnostic windows); the run refuses to start until they are set.
