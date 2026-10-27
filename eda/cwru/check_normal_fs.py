"""Diagnostic: sampling rate of the CWRU normal-baseline files (97-100).

Usage:
    uv run python eda/cwru/check_normal_fs.py configs/normal_fs_cwru.yaml

The data center does not state the rate of the normal files; papers disagree (12 or 48 kHz).
Test: read each file's DE channel as if sampled at 12 kHz. If the true rate is 48 kHz, every
frequency appears at 1/4 of its true value, so the shaft frequency and its harmonics show up at
1/4, 1/2 and 3/4 of the expected shaft frequency. 12 kHz fault files are included as controls.
For each fraction r: Welch spectrum's highest value within +-peak_rel of r * shaft, in dB above
the median within +-floor_rel. Writes runs/<date_time>_<name>/normal_fs.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import welch

from harness import runs


def main(config_path):
    cfg = runs.load_config(config_path)
    folder = runs.new_run_dir(config_path, cfg)
    rows = []
    for n, rpm_table in cfg["files"].items():
        m = scipy.io.loadmat(os.path.join(cfg["source"], f"{n}.mat"))
        x = m[f"X{n:03d}_DE_time"].ravel()
        # speed stored in the file if present, otherwise the approximate speed from the data center's table
        key = f"X{n:03d}RPM"
        rpm = float(m[key].ravel()[0]) if key in m else float(rpm_table)
        f, p = welch(x - x.mean(), fs=cfg["assumed_fs_hz"], nperseg=min(cfg["welch_nperseg"], len(x)))
        for r in cfg["fractions"]:
            f0 = r * rpm / 60
            peak = p[np.abs(f - f0) <= cfg["peak_rel"] * f0].max()
            floor = np.median(p[np.abs(f - f0) <= cfg["floor_rel"] * f0])
            rows.append({"file": n, "n_samples": len(x), "rpm": rpm, "rpm_from_file": key in m,
                         "fraction_of_shaft": r, "db_above_floor": 10 * np.log10(peak / floor)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(folder, "normal_fs.csv"), index=False)
    print(out.pivot_table(index="file", columns="fraction_of_shaft", values="db_above_floor").round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
