"""Synthetic bearing-like data with known factors (METHOD_IDEAS M5; D196).

A toy model for quick tests of the split representation (M1) and cross-modal ideas (M3).
It says nothing about real lubrication physics. Every setting comes from one config
(configs/synthetic/*.yaml); nothing has a default here.

Factors of each run (one run = one capture):
  speed (rpm)       -> periodic lines at shaft harmonics and one characteristic order
  temperature (C)   -> broadband noise level
  state (class)     -> burst process: rate and amplitude of short decaying sine bursts
  bearing (ID)      -> per-run gain and resonance filter, drawn per sensor (private nuisance)
Two sensors, AE-like and US-like, on one stream: they share speed, temperature, state and
the burst times; carrier frequency, weights and bearing nuisance differ per sensor.

A kappa-like number kappa_like = kappa_scale * (rpm / rpm_ref)^a * (temp_ref / temp)^b is
stored for every run; nominal_state = its class (bins). Designs:
  crossed     : every speed x temperature x state x bearing combination (state independent of the operating point)
  confounded  : state = nominal_state (as kappa on the current Herning run)
  departures  : as confounded, but in a share of the runs state is another class (fixed operating point, other state)
"""

import json

import numpy as np
import pandas as pd
from scipy import signal

STREAM = "acoustic"
SENSORS = ("AE", "US")


def kappa_like(rpm, temp, k):
    """The kappa-like number of an operating point (k: the 'kappa_like' section of the config)."""
    return k["scale"] * (rpm / k["rpm_ref"]) ** k["a"] * (k["temp_ref"] / temp) ** k["b"]


def run_table(cfg):
    """One dict per run: its factors. Order and states depend only on the config and its seed."""
    rng = np.random.default_rng([cfg["seed"], 0])
    n_states = len(cfg["states"]["burst_rate_hz"])
    rows = []
    for rpm in cfg["speeds_rpm"]:
        for temp in cfg["temperatures_c"]:
            kl = kappa_like(rpm, temp, cfg["kappa_like"])
            nominal = int(np.digitize(kl, cfg["kappa_like"]["state_bins"]))  # class of the operating point
            for bearing in range(cfg["n_bearings"]):
                for rep in range(cfg["repetitions"]):
                    if cfg["design"] == "crossed":
                        states, departure = range(n_states), False
                    elif cfg["design"] == "confounded":
                        states, departure = [nominal], False
                    elif cfg["design"] == "departures":
                        departure = rng.random() < cfg["departure_fraction"]
                        others = [s for s in range(n_states) if s != nominal]
                        states = [int(rng.choice(others))] if departure else [nominal]
                    else:
                        raise ValueError(f"unknown design {cfg['design']}")
                    for state in states:
                        rows.append({"speed_rpm": float(rpm), "temperature_c": float(temp), "state": int(state),
                                     "nominal_state": nominal, "departure": bool(departure),
                                     "bearing": f"B{bearing}", "repetition": rep, "kappa_like": float(kl)})
    for i, r in enumerate(rows):
        r["run_id"] = f"r{i:04d}"
    return rows


def bearing_nuisance(cfg):
    """Gain and resonance per bearing and sensor, drawn once (the same bearing has the same nuisance in every run)."""
    rng = np.random.default_rng([cfg["seed"], 1])
    b = cfg["bearing"]
    return {(f"B{i}", s): {"gain": float(np.exp(rng.normal(0.0, b["log_gain_sd"]))),
                           "resonance_hz": float(rng.uniform(*b["resonance_hz"][s]))}
            for i in range(cfg["n_bearings"]) for s in SENSORS}


def generate_run(cfg, run, nuisance, index):
    """The two sensor signals of one run, float32 (n_samples,) each."""
    rng = np.random.default_rng([cfg["seed"], 2, index])
    fs, n = cfg["sample_rate_hz"], cfg["n_samples"]
    t = np.arange(n) / fs
    f_shaft = run["speed_rpm"] / 60.0

    # Burst times are shared by the two sensors (the same contact events); Poisson process.
    rate = cfg["states"]["burst_rate_hz"][run["state"]]
    n_bursts = rng.poisson(rate * n / fs)
    starts = rng.integers(0, n, size=n_bursts)
    log_amps = rng.normal(np.log(cfg["states"]["burst_amp"][run["state"]]), cfg["states"]["burst_log_amp_sd"],
                          size=n_bursts)

    out = {}
    for s in SENSORS:
        w = cfg["sensors"][s]
        phases = rng.uniform(0, 2 * np.pi, size=cfg["periodic"]["n_harmonics"] + 1)
        x = np.zeros(n)
        # periodic part: harmonics k * f_shaft with amplitude amp / k, plus one characteristic order
        for k in range(1, cfg["periodic"]["n_harmonics"] + 1):
            x += w["periodic_weight"] * cfg["periodic"]["amp"] / k * np.sin(2 * np.pi * k * f_shaft * t + phases[k])
        x += (w["periodic_weight"] * cfg["periodic"]["order_amp"]
              * np.sin(2 * np.pi * cfg["periodic"]["order"] * f_shaft * t + phases[0]))
        # bursts: decaying sine at the sensor's carrier frequency, starting at the shared times
        tau, fc = w["burst_decay_s"], w["burst_carrier_hz"]
        length = min(n, int(np.ceil(cfg["burst_cutoff_decays"] * tau * fs)))
        tb = np.arange(length) / fs
        shape = np.exp(-tb / tau) * np.sin(2 * np.pi * fc * tb)
        for start, la in zip(starts, log_amps):
            stop = min(n, start + length)
            x[start:stop] += w["burst_weight"] * np.exp(la) * shape[:stop - start]
        # broadband noise, level set by temperature
        level = cfg["noise_std_per_temperature"][cfg["temperatures_c"].index(run["temperature_c"])]
        x += w["noise_weight"] * level * rng.normal(size=n)
        # bearing nuisance: gain, plus the signal passed through a resonance added on top
        nu = nuisance[(run["bearing"], s)]
        b, a = signal.iirpeak(nu["resonance_hz"], cfg["bearing"]["resonance_q"], fs=fs)
        x = nu["gain"] * (x + cfg["bearing"]["resonance_mix"] * signal.lfilter(b, a, x))
        out[s] = x.astype(np.float32)
    return out


def read(cfg, run, nuisance, index):
    """(tables, signals) of one run for canonical.write_run (D4 adapter output)."""
    rid, capture_id = run["run_id"], "capture_0"
    runs = pd.DataFrame([{
        "dataset": cfg["dataset"], "run_id": rid, "status": "final",  # generated, valid by construction (D7)
        "bearing": run["bearing"], "lubricant": None, "source_name": "generated",
        "source_size_bytes": 0, "source_mtime_utc": None,
        "extras": json.dumps({"generator": "harness/data/synthetic.py", "config": cfg,
                              "nuisance": {s: nuisance[(run["bearing"], s)] for s in SENSORS}}),
    }])
    captures = pd.DataFrame([{
        "run_id": rid, "capture_id": capture_id, "stream": STREAM, "source_name": "generated",
        "start_s": np.nan, "n_samples": cfg["n_samples"], "timing_accuracy": "not applicable",
        **{k: run[k] for k in ("speed_rpm", "temperature_c", "state", "nominal_state", "departure",
                               "bearing", "repetition", "kappa_like")},
    }])
    channels = pd.DataFrame([{
        "run_id": rid, "stream": STREAM, "channel": s, "sensor_type": f"synthetic {s}-like",
        "sample_rate_hz": float(cfg["sample_rate_hz"]), "units": "arbitrary", "scale": 1.0, "offset": 0.0,
        "x_origin_s": np.nan, "dtype": "float32",
    } for s in SENSORS])
    telemetry = pd.DataFrame([{"run_id": rid, "capture_id": capture_id}])  # the factors are on the captures table
    values = generate_run(cfg, run, nuisance, index)
    signals = [(capture_id, s, values[s], 1.0, 0.0, "arbitrary") for s in SENSORS]
    return {"runs": runs, "captures": captures, "channels": channels, "telemetry": telemetry}, signals
