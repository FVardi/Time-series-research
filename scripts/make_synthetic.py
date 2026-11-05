"""Generate a synthetic dataset in the canonical format (METHOD_IDEAS M5, D196): one run per factor combination.

Usage:
    uv run python scripts/make_synthetic.py configs/synthetic/crossed.yaml data/canonical

The same config always gives the same data (seeded). Canonical data is never overwritten (D5):
delete data/canonical/<dataset> by hand to regenerate.
"""

import sys

import yaml

from harness.data import canonical, synthetic


def make(config_path, root):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    runs = synthetic.run_table(cfg)
    nuisance = synthetic.bearing_nuisance(cfg)
    for i, run in enumerate(runs):
        tables, signals = synthetic.read(cfg, run, nuisance, i)
        canonical.write_run(root, cfg["dataset"], run["run_id"], tables, signals)
    print(f"Done: {len(runs)} runs of {cfg['dataset']} in {root}.")


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
