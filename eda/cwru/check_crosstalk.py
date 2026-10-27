"""Diagnostic: does a bearing fault at one end of the CWRU motor show in the other end's channel?

Usage:
    uv run python eda/cwru/check_crosstalk.py configs/crosstalk_cwru.yaml

Each 12 kHz fault file holds a drive-end (DE) and a fan-end (FE) channel. For every fault file,
the squared envelope spectrum (high-pass, Hilbert envelope squared, |FFT|) of
  own       - the channel at the faulty bearing's end,
  opposite  - the channel at the other end (it measures a healthy bearing),
  reference - the same opposite channel in the normal file at the same load (no fault anywhere),
is read at harmonics k of the faulty bearing's defect frequency (inner race: BPFI, outer race:
BPFO, ball: BSF; multiples of shaft speed from the data center's bearing information page).
Value: highest value within +-peak_rel of k * f_defect, in dB above the median within +-floor_rel
(peak window excluded). Cross-talk shows as opposite clearly above reference.
The normal files are sampled at 48 kHz (check_normal_fs.py); the reference is resampled to 12 kHz
(polyphase, factor 1/4) so that both pass through the same filter band.
Writes runs/<date_time>_<name>/crosstalk.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from harness import runs

DEFECT = {"inner": "bpfi", "outer": "bpfo", "ball": "bsf"}  # fault location -> defect frequency
OTHER_END = {"DE": "FE", "FE": "DE"}


def read_channel(cfg, n, channel):
    """One channel of a .mat file and the speed (from the file, else the table speed of its load)."""
    m = scipy.io.loadmat(os.path.join(cfg["source"], f"{n}.mat"))
    x = m[f"X{n:03d}_{channel}_time"].ravel()
    key = f"X{n:03d}RPM"
    return x, (float(m[key].ravel()[0]) if key in m else None)


def envelope_spectrum(x, fs, cfg):
    """Squared envelope spectrum (amplitude) after a zero-phase Butterworth high-pass."""
    sos = butter(cfg["filter_order"], cfg["highpass_hz"], btype="highpass", fs=fs, output="sos")
    env2 = np.abs(hilbert(sosfiltfilt(sos, x - x.mean()))) ** 2
    return np.fft.rfftfreq(len(env2), 1 / fs), np.abs(np.fft.rfft(env2 - env2.mean()))


def db_at(freqs, spec, f0, cfg):
    """Peak near f0 in dB above the local median (peak window excluded from the median)."""
    near = np.abs(freqs - f0)
    peak = spec[near <= cfg["peak_rel"] * f0].max()
    floor = np.median(spec[(near <= cfg["floor_rel"] * f0) & (near > cfg["peak_rel"] * f0)])
    return 20 * np.log10(peak / floor)


def main(config_path):
    cfg = runs.load_config(config_path)
    folder = runs.new_run_dir(config_path, cfg)
    fs = cfg["fs_hz"]
    # normal-file reference spectra per (load, channel), resampled to the fault files' rate
    ref = {}
    for load, n in cfg["normal_files"].items():
        for ch in ("DE", "FE"):
            x, _ = read_channel(cfg, n, ch)
            x = resample_poly(x, 1, cfg["normal_fs_hz"] // fs)
            ref[(load, ch)] = envelope_spectrum(x, fs, cfg)
    rows = []
    for fault, spec in cfg["faults"].items():
        end, location = spec["end"], spec["location"]
        mult = cfg["defect_multiples"][end][DEFECT[location]]
        for load, n in enumerate(spec["files"]):
            if n is None:
                continue
            try:
                own_x, rpm = read_channel(cfg, n, end)
                opp_x, _ = read_channel(cfg, n, OTHER_END[end])
            except Exception as e:  # e.g. an incomplete download: reported, not hidden
                print(f"skipped file {n} ({fault}, load {load}): {type(e).__name__}")
                continue
            if rpm is None:
                rpm = cfg["table_rpm"][load]
            own, opp = envelope_spectrum(own_x, fs, cfg), envelope_spectrum(opp_x, fs, cfg)
            for k in cfg["harmonics"]:
                f0 = k * mult * rpm / 60
                rows.append({"fault": fault, "end": end, "location": location, "size_mil": spec["size_mil"],
                             "clock": spec.get("clock"), "load_hp": load, "file": n, "rpm": rpm,
                             "harmonic": k, "freq_hz": f0,
                             "own_db": db_at(*own, f0, cfg),
                             "opposite_db": db_at(*opp, f0, cfg),
                             "reference_db": db_at(*ref[(load, OTHER_END[end])], f0, cfg)})
    out = pd.DataFrame(rows)
    out["opposite_minus_reference_db"] = out["opposite_db"] - out["reference_db"]
    out.to_csv(os.path.join(folder, "crosstalk.csv"), index=False)
    summary = out.groupby(["end", "fault", "harmonic"])[["own_db", "opposite_db", "reference_db"]].mean()
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
