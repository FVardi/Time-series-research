# FreRA

**Source:** Tian, Miao & Qian, KDD 2025; official code github.com/Tian0426/FreRA (SimCLR
framework), submodule `external/frera` at 7236fbf. Wrapped (D10 mode A): adapter
`harness/methods/frera.py` runs main_worker's steps with the official `FreRA`, `SimCLR`,
`adjust_learning_rate` and `train` per epoch (D109).
**Reported as:** FreRA. **Settings:** `configs/methods/frera.yaml`.

## Current rules
- README commands: batch 128, 200 epochs, SGD lr 0.01; temperature 0.2 (paper), projector 128;
  frequency weights: lr 0.001, Gumbel temperature 0.1 (D109).
- Regularisation weight `l1_weight` 0.1, the script's default (README: 0.003) (D112).
- Sized from the data on every dataset; no setting depends on the input shape (D158, D173).
- Saves only the encoder and the frequency weights (D237).
- Not run on Paderborn and XJTU-SY: the official projector is Linear(d, d) on the flattened FCN
  output, d = 128 × 642 = 82,176 on 5,120 samples (6.75 B weights; out of memory on 24 GB), more
  on XJTU-SY's 32,768 samples (D264). Runs on FordA, CWRU, HUST, Herning.
- Run fixes: `.cuda()` in `get_sampling` replaced by the weights' device; FCN output length set
  from the series length; fitlog and data_loaders imports stubbed (D109).

## Findings
- The code's 0.1 corresponds to λ ≈ 0.05 in the paper's scale, below its search range; the README
  commands are probably not the settings behind the paper's results (D112).

## Open
- FordA value in the paper's Table 8 not yet read (paper protocol: best test accuracy).

History: D109, D112, D158, D173, D237, D256, D264.
