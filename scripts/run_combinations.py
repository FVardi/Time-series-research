"""Run every combination of a config's literature protocol and report the mean over them (D220).

Usage:
    uv run python scripts/run_combinations.py configs/fft_paderborn.yaml
    uv run python scripts/run_combinations.py configs/fft_paderborn.yaml --seeds 0 --combinations 0 1

For each combination k: scripts/pretrain.py --combination k, then scripts/evaluate.py on its run.
Then writes runs/<date_time>_<name>_combinations.csv with the test accuracy of every combination,
probe, label fraction and seed, and prints the mean over the combinations per probe, label fraction
and seed: Lessmeier et al. (2016) report "the mean classification accuracy for the 10 combinations".
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import torch

import evaluate  # scripts/evaluate.py (this script's folder is on the import path)
import pretrain  # scripts/pretrain.py
from harness import protocols, runs

N_COMBINATIONS = {"paderborn_lessmeier": len(protocols.COMBINATIONS)}  # 10 (D220)


def main(config_path, seeds=None, combinations=None):
    cfg = runs.load_config(config_path)
    protocol = cfg["split"]["protocol"]
    combinations = combinations if combinations is not None else range(N_COMBINATIONS[protocol])
    rows = []
    for k in combinations:
        folder = pretrain.main(config_path, seeds, k)
        evaluate.main(folder)
        results = pd.read_csv(os.path.join(folder, "results.csv"))
        rows.append(results.assign(combination=k, run=folder.replace(os.sep, "/")))
    table = pd.concat(rows, ignore_index=True)
    out = os.path.join("runs", f"{datetime.now():%Y%m%d_%H%M%S}_{cfg['name']}_combinations.csv")
    table.to_csv(out, index=False)
    mean = table.groupby(["probe", "label_fraction", "seed"]).accuracy.agg(["mean", "count"])
    print(f"\nMean test accuracy over the combinations (count = combinations):\n{mean}\n\nAll rows: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seeds", type=int, nargs="+", help="seeds instead of the config's (D201)")
    parser.add_argument("--combinations", type=int, nargs="+", help="only these combinations (default: all)")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    main(args.config, args.seeds, args.combinations)
