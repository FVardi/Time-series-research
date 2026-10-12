"""D52: per-band check of where the Herning signals carry information.

Usage:
    uv run python scripts/check_bands.py configs/band_check_herning.yaml

Reads the source file sweep by sweep (no conversion needed). For every sweep and
channel, the Welch power spectral density is integrated over each band. Writes into
runs/<date_time>_<name>/:
    sweeps.csv         one row per sweep: rpm, temperature, kappa, role
                       (analysed / noise_reference / non_rotating_0 / excluded)
    band_power.csv     one row per sweep, channel and band: band power (V^2)
    summary.csv        per channel and band: median power of analysed and
                       noise-reference sweeps, their ratio in dB, and the Spearman
                       correlation of band power with rpm, temperature and kappa
"""

import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

from harness import kappa, runs


def bands(cfg_s, nyquist):
    """[(label, f_lo, f_hi)]: one low band, then base-10 third-octave bands up to Nyquist."""
    lo, hi = cfg_s["low_band_hz"]
    out = [(f"{lo:g}-{hi:g} Hz", lo, hi)]
    G = 10 ** 0.3
    x = round(3 * np.log10(cfg_s["third_octave_from_hz"] / 1000) / 0.3)
    while True:
        fm = 1000 * G ** (x / 3)
        f1, f2 = fm * G ** (-1 / 6), min(fm * G ** (1 / 6), nyquist)
        if f1 >= nyquist:
            return out
        out.append((f"{fm / 1e3:.4g} kHz", f1, f2))
        x += 1


def role(rpm_target, temp_target, rpm, temp, op):
    """Which group a sweep belongs to (D52, D55)."""
    if temp_target == op["cooldown_temp_target"]:
        return "excluded"
    if rpm_target == op["non_rotating_rpm_target"]:
        return "noise_reference"
    if rpm_target == 0:
        return "non_rotating_0"  # target 0: not rotating either; reported separately
    if op["rpm_min"] <= rpm <= op["rpm_max"] and temp >= op["temp_min"]:
        return "analysed"
    return "excluded"


def sweep_order(f):
    """Source sweep names in acquisition order."""
    sweeps = f["sweeps"]
    return sorted(sweeps.keys(), key=lambda n: sweeps[n].attrs["tick"])


def process(cfg, f, names, first_index=0):
    """Operating point, role and band powers for the given sweeps. Returns (sweep rows, power rows)."""
    op, sp = cfg["operating_point"], cfg["spectrum"]
    sweeps = f["sweeps"]
    fs = float(sweeps[names[0]][cfg["channels"][0]].attrs["sample_rate"])
    band_list = bands(sp, fs / 2)
    sweep_rows, power_rows = [], []
    for i, name in enumerate(names, start=first_index):
        a = sweeps[name].attrs
        rpm = float(a["telem_vfd_cmd_hz"]) * op["rpm_per_cmd_hz"]
        temp = float(a[op["temperature_field"]])
        r = role(float(a["telem_rpm_target"]), float(a["telem_temp_target"]), rpm, temp, op)
        k = kappa.kappa(rpm, temp, op["d_pw_mm"], op["nu_40_cst"], op["nu_100_cst"]) if r == "analysed" else np.nan
        sweep_rows.append({"sweep": name, "order": i, "rpm": rpm, "temperature_c": temp,
                           "temp_target": float(a["telem_temp_target"]), "kappa": k, "role": r})
        if r == "excluded":
            continue
        for ch in cfg["channels"]:
            x = sweeps[name][ch]["voltage"][:]
            freqs, psd = welch(x, fs=fs, window=sp["welch_window"], nperseg=sp["welch_nperseg"],
                               noverlap=sp["welch_noverlap"])
            df = freqs[1] - freqs[0]
            for label, f1, f2 in band_list:
                sel = (freqs >= f1) & (freqs < f2)
                power_rows.append({"sweep": name, "channel": ch, "band": label, "f_lo": f1, "f_hi": f2,
                                   "power": float(psd[sel].sum() * df)})
        if i % 200 == 0:
            print(f"{i} sweeps", flush=True)
    return sweep_rows, power_rows


def summarise(sweeps_df, power):
    """Per channel and band: signal vs noise reference, and dependence on the operating point."""
    p = power.merge(sweeps_df, on="sweep")
    rows = []
    for (ch, band), g in p.groupby(["channel", "band"], sort=False):
        on, off, off0 = g[g.role == "analysed"], g[g.role == "noise_reference"], g[g.role == "non_rotating_0"]
        row = {"channel": ch, "band": band, "f_lo": g.f_lo.iloc[0], "f_hi": g.f_hi.iloc[0],
               "median_power_analysed": on.power.median(), "median_power_noise_ref": off.power.median(),
               "ratio_db": 10 * np.log10(on.power.median() / off.power.median()),
               "ratio_db_vs_target0": 10 * np.log10(on.power.median() / off0.power.median()),
               "n_analysed": len(on), "n_noise_ref": len(off)}
        for var in ("rpm", "temperature_c", "kappa"):
            row[f"spearman_{var}"] = spearmanr(on.power, on[var]).statistic  # rank-based: same for log power
        rows.append(row)
    return pd.DataFrame(rows)


def write(folder, sweeps_df, power):
    """Write the three result files."""
    sweeps_df.to_csv(os.path.join(folder, "sweeps.csv"), index=False)
    power.to_csv(os.path.join(folder, "band_power.csv"), index=False)
    summary = summarise(sweeps_df, power)
    summary.to_csv(os.path.join(folder, "summary.csv"), index=False)
    print(summary.to_string())


def main(config_path):
    cfg = runs.load_config(config_path)
    folder = runs.new_run_dir(config_path, cfg)
    with h5py.File(cfg["source"], "r") as f:
        sweep_rows, power_rows = process(cfg, f, sweep_order(f))
    write(folder, pd.DataFrame(sweep_rows), pd.DataFrame(power_rows))


if __name__ == "__main__":
    main(sys.argv[1])
