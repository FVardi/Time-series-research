"""Small helpers for runs (D16, D17): config, seeds, run folders."""

import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import yaml


def load_config(path):
    """Read a YAML config. No defaults are filled in by the code: every setting is in the file
    or in the shared part files it names under 'defaults:' (D260), e.g.

        defaults: [datasets/cwru.yaml, methods/ts2vec.yaml]

    Parts (paths relative to the file) are merged in the order listed, then the file's own keys
    on top; mappings are merged key by key, any other value is replaced.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    merged = {}
    for part in cfg.pop("defaults", []):
        merged = merge(merged, load_config(os.path.join(os.path.dirname(path), part)))
    return merge(merged, cfg)


def merge(base, override):
    """Deep merge of two config dicts (D260): mappings key by key, everything else replaced."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def config_files(path):
    """The config file and the part files it uses (recursively), as (path, path relative to the file's folder)."""
    with open(path, encoding="utf-8") as f:
        parts = (yaml.safe_load(f) or {}).get("defaults", [])
    out = [(path, os.path.basename(path))]
    for part in parts:
        for p, rel in config_files(os.path.join(os.path.dirname(path), part)):
            out.append((p, os.path.join(os.path.dirname(part), rel)))
    return out


def set_seed(seed):
    """Seed Python, NumPy and PyTorch."""
    import torch  # imported here so scripts without PyTorch (e.g. check_bands.py) can use this module
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit():
    """Current commit, and whether there are uncommitted changes (then results may not be reproducible)."""
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if rev.returncode != 0:  # not inside a git repository: say so instead of recording a blank commit
        return {"commit": None, "uncommitted_changes": None, "note": "not run from a git repository"}
    status = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    return {"commit": rev.stdout.strip(), "uncommitted_changes": bool(status.strip())}


def new_run_dir(config_path, config):
    """Create runs/<date_time>_<name>/ with the config and the git commit.

    config.yaml in the run folder is the config file with its shared parts merged in (D260), so
    it holds every setting of the run in one place; the files it was merged from are copied,
    comments included, into config_sources/.
    Warns when the working tree has uncommitted changes (D99): runs meant for the report
    are made from a committed tree, because the commit alone then reproduces the code.
    The run still starts; git.json records the state either way.
    """
    folder = os.path.join("runs", f"{datetime.now():%Y%m%d_%H%M%S}_{config['name']}")
    os.makedirs(folder)
    with open(os.path.join(folder, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(f"# {config_path} with its 'defaults:' parts merged in (D260); sources in config_sources/\n")
        # flow style for lists, so e.g. 'seeds: [0, 1, 2]' stays on one line (scripts/pretrain.py edits it)
        resolved = load_config(config_path)
        resolved = {"name": resolved.pop("name"), **resolved} if "name" in resolved else resolved  # name first
        yaml.safe_dump(resolved, f, sort_keys=False, default_flow_style=None, width=120)
    for source, rel in config_files(config_path):
        os.makedirs(os.path.dirname(os.path.join(folder, "config_sources", rel)), exist_ok=True)
        shutil.copy(source, os.path.join(folder, "config_sources", rel))
    git = git_commit()
    save_json(os.path.join(folder, "git.json"), git)
    if git["uncommitted_changes"]:
        print(UNCOMMITTED_WARNING, file=sys.stderr, flush=True)
    return folder


UNCOMMITTED_WARNING = (
    "\n*** WARNING: the working tree has uncommitted changes. ***\n"
    "The run is recorded with commit + 'uncommitted_changes: true' (git.json), so the commit alone\n"
    "does not reproduce the code. Runs meant for the report should start from a committed tree (D99).\n")


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def check_no_overlap(windows):
    """Stop if any capture is used both for training (incl. pretraining) and for testing (D33).

    windows: the run's window table, with columns split, run_id and capture_id.
    Captures are compared as (run_id, capture_id), because capture IDs are only unique within a run.
    """
    def captures(split):
        rows = windows[windows.split == split]
        return set(zip(rows.run_id, rows.capture_id))

    shared = captures("train") & captures("test")
    if shared:
        raise ValueError(f"{len(shared)} captures are in both training and test, e.g. {sorted(shared)[:5]}")
