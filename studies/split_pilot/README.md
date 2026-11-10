# Split-representation pilot (METHOD_IDEAS M1; D182-D185, D196-D198)

Does a block of the representation that is pushed to be free of the operating point help?
TS2Vec backbone (`harness/methods/ts2vec_split.py`), on synthetic data (D196) and CWRU.

Step 0 (done): `load_probe.py` — is the CWRU load recoverable from existing representations?

    uv run python studies/split_pilot/load_probe.py runs/<run> [runs/<run> ...]

Runs: configs in `configs/` (written once by `make_configs.py`), one per variant and split:

    uv run python scripts/pretrain.py studies/split_pilot/configs/pilot_B1_synthetic_departures.yaml
    uv run python scripts/evaluate.py runs/<the folder it prints>

Variants: 0 = plain TS2Vec at batch 64; A = split + op heads; B = A + adversary; C = A + HSIC
(B, C at removal weights 0.1, 1, 10). Splits: synthetic_random, synthetic_departures,
synthetic_departures_nominalpre, synthetic_loso{600,1200,1800,2400} (D199), cwru, cwru_bearing.

Synthetic runs in one go (D204: A, B1, C1, seed 0 only, by default):

    powershell -ExecutionPolicy Bypass -File studies\split_pilot\run_synthetic.ps1

Block probes (D205): what each part of the representation contains, for all finished runs:

    uv run --no-sync python studies/split_pilot/block_probes.py "runs/*_pilot_*" --seeds 0

No MLP probe (D209). The pilot is parked (D215); results and next options in METHOD_IDEAS M1.
