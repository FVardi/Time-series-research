# ts2vec_split (own method, pilot — parked)

Split-representation pilot of METHOD_IDEAS M1: TS2Vec's encoder, cropping and loss in an own loop,
the representation split into `z_op` (operating point, 32 dims) and `z_state` (288 dims), one head
per operating-condition factor on `z_op`, and removal of the factors from `z_state` by an adversary
(gradient reversal, DANN ramp) or HSIC (D182, D183, D197). Code `harness/methods/ts2vec_split.py`;
study `studies/split_pilot/` (configs, block probes, results).

**Status: parked** (D215): on the synthetic data `z_state` keeps the operating point under every
removal tried (adversary at weight 1, HSIC at 1, 100, 1000), and large HSIC weights break training.
Results and the options for picking it up again: METHOD_IDEAS.md M1.

Rules while it existed: conditions passed through the `conditions:` config key (D198); logreg
block probes only, seed 0 (D205, D206, D209); batch 64, 600 iterations (D185).

History: D182–D185, D196–D199, D204–D207, D209, D215.
