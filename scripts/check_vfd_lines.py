"""Diagnostic: height of the ~4.67 kHz comb lines with the frequency converter off and on (QUESTIONS.md, Q3).

Usage:
    uv run python scripts/check_vfd_lines.py configs/vfd_lines_herning.yaml

Groups (same temperature target): 'vfd_off' = rpm target 0 (converter not running),
'vfd_on_not_rotating' = rpm target 100 (converter running, bearing confirmed not rotating).
For each group, channel and harmonic k: the mean Welch spectrum's highest value within
+-line_halfwidth of k * line_spacing, in dB above the median within +-floor_halfwidth.
Writes runs/<date_time>_<name>/lines.csv.
"""

import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

from harness import runs

GROUPS = {"vfd_off": 0.0, "vfd_on_not_rotating": 100.0}  # name: rpm target


def main(config_path):
    cfg = runs.load_config(config_path)
    folder = runs.new_run_dir(config_path, cfg)
    rows = []
    with h5py.File(cfg["source"], "r") as f:
        sweeps = f["sweeps"]
        for group, rpm_target in GROUPS.items():
            names = [n for n in sweeps if float(sweeps[n].attrs["telem_rpm_target"]) == rpm_target
                     and float(sweeps[n].attrs["telem_temp_target"]) == cfg["temp_target"]][:cfg["n_sweeps"]]
            running = sorted({bool(sweeps[n].attrs["telem_vfd_is_running"]) for n in names})
            for ch in cfg["channels"]:
                psd = []
                for n in names:
                    fs = float(sweeps[n][ch].attrs["sample_rate"])
                    freqs, p = welch(sweeps[n][ch]["voltage"][:], fs=fs, window=cfg["welch_window"],
                                     nperseg=cfg["welch_nperseg"], noverlap=cfg["welch_noverlap"])
                    psd.append(p)
                psd = np.mean(psd, axis=0)
                for k in range(cfg["harmonics"][0], cfg["harmonics"][1] + 1):
                    f0 = k * cfg["line_spacing_hz"]
                    line = psd[np.abs(freqs - f0) <= cfg["line_halfwidth_hz"]].max()
                    floor = np.median(psd[np.abs(freqs - f0) <= cfg["floor_halfwidth_hz"]])
                    rows.append({"group": group, "vfd_is_running": running, "channel": ch, "harmonic": k,
                                 "freq_hz": f0, "line_db_above_floor": 10 * np.log10(line / floor),
                                 "n_sweeps": len(names)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(folder, "lines.csv"), index=False)
    print(out.pivot_table(index="harmonic", columns=["channel", "group"], values="line_db_above_floor").round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
