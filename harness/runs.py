"""Small helpers for runs (D16, D17): config, seeds, run folders."""

import json
import os
import random
import shutil
import subprocess
from datetime import datetime

import numpy as np
import torch
import yaml


def load_config(path):
    """Read a YAML config. No defaults are filled in: every setting must be in the file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Seed Python, NumPy and PyTorch."""
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
    """Create runs/<date_time>_<name>/ with a copy of the config and the git commit."""
    folder = os.path.join("runs", f"{datetime.now():%Y%m%d_%H%M%S}_{config['name']}")
    os.makedirs(folder)
    shutil.copy(config_path, os.path.join(folder, "config.yaml"))
    save_json(os.path.join(folder, "git.json"), git_commit())
    return folder


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
