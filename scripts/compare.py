"""Comparison of methods across runs (D203): one set of tables from many runs' results.csv.

Usage:
    uv run python scripts/compare.py configs/compare/forda.yaml

The comparison config lists, per setting (a dataset with its split, e.g. "FordA" or
"CWRU, bearing split"), which run folder counts for each method and whether it is a
baseline. Runs are chosen by hand, because runs/ also holds superseded and failed runs.
Each listed run must have been evaluated (scripts/evaluate.py) first.

Before comparing, the script stops unless the runs of a setting can be paired (D78): same
dataset and task, the same windows (windows.csv, where the run has one) and, per seed,
the same training and test labels in the same order.

Writes runs/<date_time>_<name>/ with a copy of the config and:
  long.csv     every row of every listed run's results.csv, with setting, method, role and run
  summary.csv  mean, SD and number of seeds per setting, method, probe and label fraction;
               a cell a method has not run is listed with n_seeds = 0 (D13: missing, not dropped)
  paired.csv   per setting, probe, label fraction and metric: the best baseline (highest mean
               test score, lowest for MAE; D80) and each method's difference to it per seed
               (D78), signed so that positive = the method is better. Probes that ignore the
               features (floor, floor_mean; D83) are not paired.
  timing.csv   per setting, method and seed: the time of each stage (D79); empty for runs
               made before timing was recorded
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from harness import runs

ROLES = ("method", "baseline")
# Metrics per task (D26 classes, D59 regression) and whether higher is better
METRICS = {"classification": {"accuracy": True, "macro_f1": True}, "regression": {"mae": False, "r2": True}}
TIMING = ["pretrain_s", "encode_train_s", "encode_test_s", "encode_series_per_s"]
NO_FEATURES = ("floor", "floor_mean")  # probes that ignore the features (D83): nothing to pair
WINDOW_COLUMNS = ["split", "row", "run_id", "capture_id", "start", "stop"]  # what identifies a window (D33)


def load_run(setting, method, entry):
    """The run's results.csv with setting, method, role and run added, and its timing per seed."""
    if entry["role"] not in ROLES:
        raise ValueError(f"{setting}/{method}: role must be one of {ROLES}, not {entry['role']!r}")
    path = os.path.join(entry["run"], "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run scripts/evaluate.py on {entry['run']} first")
    results = pd.read_csv(path).assign(setting=setting, method=method, role=entry["role"], run=entry["run"])
    timing = []
    for seed in sorted(results.seed.unique()):
        t_path = os.path.join(entry["run"], f"seed_{seed}", "timing.json")
        t = json.load(open(t_path)) if os.path.exists(t_path) else {}  # runs before D79 have no timing
        timing.append({"setting": setting, "method": method, "seed": seed, **{k: t.get(k) for k in TIMING}})
    return results, timing


def check_pairable(setting, entries, seeds):
    """Stop unless all runs of the setting can be paired (D78); returns the setting's task.

    entries: {method: {run, role}}; seeds: {method: seeds in its results.csv}.
    """
    cfgs = {m: runs.load_config(os.path.join(e["run"], "config.yaml")) for m, e in entries.items()}
    for key in ("dataset", "task"):
        values = {m: c.get(key) for m, c in cfgs.items()}  # task absent = classification
        if len(set(values.values())) > 1:
            raise ValueError(f"{setting}: the runs differ in '{key}': {values}")
    # Same windows, compared on the columns that identify a window
    tables = {}
    for m, e in entries.items():
        path = os.path.join(e["run"], "windows.csv")
        if os.path.exists(path):
            tables[m] = pd.read_csv(path)[WINDOW_COLUMNS]
        else:
            print(f"note: {setting}/{m} has no windows.csv (made before D33); compared on its labels only")
    for m, t in tables.items():
        first = next(iter(tables))
        if not t.equals(tables[first]):
            raise ValueError(f"{setting}: {m} and {first} use different windows (windows.csv)")
    # Same labels per seed, in the same order, so the label subsets are the same (D29, D78)
    for seed in sorted({s for ss in seeds.values() for s in ss}):
        having = [m for m in entries if seed in seeds[m]]
        for m in having[1:]:
            for name in ("y_train.npy", "y_test.npy"):
                a = np.load(os.path.join(entries[having[0]]["run"], f"seed_{seed}", name))
                b = np.load(os.path.join(entries[m]["run"], f"seed_{seed}", name))
                if not np.array_equal(a, b):
                    raise ValueError(f"{setting}, seed {seed}: {m} and {having[0]} differ in {name}")
    task = next(iter(cfgs.values())).get("task")
    return "regression" if task == "regression" else "classification"


def summarise(long):
    """Mean, SD and number of seeds per cell; cells a method has not run get n_seeds = 0 (D13)."""
    keys = ["setting", "method", "role", "probe", "label_fraction"]
    cols = [c for m in METRICS.values() for c in m if c in long] + (["probe_fit_s"] if "probe_fit_s" in long else [])
    g = long.groupby(keys, sort=False)
    summary = g[cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    summary["n_seeds"] = g.seed.nunique()
    # Full grid per setting: every method x every (probe, label fraction) any run of that setting has
    grid = []
    for setting, part in long.groupby("setting", sort=False):
        methods = part[["method", "role"]].drop_duplicates()
        cells = part[["probe", "label_fraction"]].drop_duplicates()
        grid.append(methods.merge(cells, how="cross").assign(setting=setting))
    summary = pd.concat(grid)[keys].merge(summary.reset_index(), how="left", on=keys)
    summary["n_seeds"] = summary.n_seeds.fillna(0).astype(int)  # 0 = not run
    return summary


def per_seed(long, setting, method, probe, fraction, metric):
    """One method's test score per seed in one cell, indexed by seed."""
    rows = long[(long.setting == setting) & (long.method == method) & (long.probe == probe)
                & (long.label_fraction == fraction)]
    return rows.set_index("seed")[metric]


def paired(long, summary, tasks):
    """Each method's difference to the best baseline per seed (D78, D80); positive = the method is better."""
    rows = []
    done = summary[(summary.n_seeds > 0) & ~summary.probe.isin(NO_FEATURES)]
    for (setting, probe, fraction), cell in done.groupby(["setting", "probe", "label_fraction"], sort=False):
        base = cell[cell.role == "baseline"]
        if base.empty:  # no baseline has this probe and budget: nothing to pair with
            continue
        for metric, higher_better in METRICS[tasks[setting]].items():
            sign = 1 if higher_better else -1
            # D80: the best baseline by mean test score (favours the baselines); on a tie, the first listed
            best = base.loc[(sign * base[f"{metric}_mean"]).idxmax()]
            ref = per_seed(long, setting, best.method, probe, fraction, metric)
            for method in cell[cell.role == "method"].method:
                diff = sign * (per_seed(long, setting, method, probe, fraction, metric) - ref)
                for seed, value in diff.dropna().items():  # only seeds both runs have
                    rows.append({"setting": setting, "probe": probe, "label_fraction": fraction, "metric": metric,
                                 "best_baseline": best.method, "best_baseline_mean": best[f"{metric}_mean"],
                                 "method": method, "seed": seed, "difference": value})
    return pd.DataFrame(rows)


def main(config_path):
    cfg = runs.load_config(config_path)
    parts, timing, tasks = [], [], {}
    for setting, entries in cfg["settings"].items():
        seeds = {}
        for method, entry in entries.items():
            results, t = load_run(setting, method, entry)
            parts.append(results)
            timing += t
            seeds[method] = set(results.seed)
        tasks[setting] = check_pairable(setting, entries, seeds)
    long = pd.concat(parts, ignore_index=True)
    summary = summarise(long)
    diffs = paired(long, summary, tasks)

    folder = runs.new_run_dir(config_path, cfg)
    long.to_csv(os.path.join(folder, "long.csv"), index=False)
    summary.to_csv(os.path.join(folder, "summary.csv"), index=False)
    diffs.to_csv(os.path.join(folder, "paired.csv"), index=False)
    pd.DataFrame(timing).to_csv(os.path.join(folder, "timing.csv"), index=False)

    # On screen: mean test score per method and cell, and the mean paired difference
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        for setting in cfg["settings"]:
            for metric in METRICS[tasks[setting]]:
                s = summary[summary.setting == setting]
                print(f"\n{setting}: {metric}, mean over seeds (NaN = not run)")
                print(s.pivot_table(index="method", columns=["probe", "label_fraction"], values=f"{metric}_mean",
                                    sort=False, dropna=False))
                if len(diffs):
                    d = diffs[(diffs.setting == setting) & (diffs.metric == metric)]
                    if len(d):
                        print(f"{setting}: {metric}, mean paired difference to the best baseline (positive = better)")
                        print(d.pivot_table(index="method", columns=["probe", "label_fraction"],
                                            values="difference", sort=False))
    print(f"\nDone: {folder}")


if __name__ == "__main__":
    main(sys.argv[1])
