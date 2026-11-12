# studies/method_checks — one-off checks of single methods

Status: reference scripts; no open work.

Small scripts that checked one property of one method, kept so the checks can be repeated
(moved here from `scripts/` by D262; run from the repository root):

| script | what | decisions |
|---|---|---|
| `reencode_patchtst.py` | re-encodes saved PatchTST models with another pooling (mean, flatten) and writes a new run folder for `scripts/evaluate.py` | D86, D87 |
| `time_tsjepa.py` | times TS-JEPA epochs for given configs (to plan the learning-rate search; later for TS-JEPA's epochs) | D66, D256 |

    uv run python studies/method_checks/reencode_patchtst.py runs/<source run> mean
    uv run python studies/method_checks/time_tsjepa.py configs/tsjepa_forda_official_masking.yaml configs/tsjepa_forda.yaml
