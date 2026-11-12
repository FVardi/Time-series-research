# studies/tsjepa_epochs — does TS-JEPA need 5,000 epochs? (D200)

Status: deferred (D200); needs the GPU to itself.

TS-JEPA is the slowest method (about 195 min per seed on the main CWRU task) because of its
5,000 epochs (D66). Its loss cannot tell when the representation stops changing (on CWRU it
drops within 10 epochs, rises to about 0.5 by epoch 500 and drifts down to about 0.41), so this
study measures the representation directly while TS-JEPA trains.

    uv run python studies/tsjepa_epochs/check_epochs.py studies/tsjepa_epochs/configs/epochs_cwru.yaml

Training is exactly `configs/tsjepa_cwru.yaml`'s seed 0 (the CWRU literature protocol, D218;
until 2026-11-05 the main task, whose config was deleted by D240) (`harness/methods/tsjepa.py` with its
`on_epoch` hook; the measurements leave the random number generators as they were, tested in
`tests/test_tsjepa.py`). Before the first epoch and every 250 epochs:

| column | meaning |
|---|---|
| `epoch` | epochs completed |
| `rankme` | RankMe of the representations of all training windows (as D152) |
| `val_accuracy` | standard logreg probe, all labels, fitted on 80 % of the training windows, scored on the other 20 % (stratified by class, one fixed draw); test windows never used |
| `measure_s` | seconds spent measuring (not counted in `timing.json`'s `pretrain_s`) |

Run it with the GPU to itself, or the times mean nothing. On the literature protocol (900 training
windows) an epoch is about five times shorter than on the main task (about 3¼ hours for 5,000 epochs).
Whether to shorten TS-JEPA's schedule is decided from the result.
