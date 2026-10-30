"""Class labels and groups, derived per dataset from the captures table (D147).

Label derivation lives here, not in the adapters (D4). Each function takes the captures
table and returns two lists, one entry per capture:
  label: the class, or None for captures without a class (e.g. unlabelled pretraining data)
  group: the unit that must stay on one side of the split, or None (D121)
"""

from harness import kappa


def cwru(captures):
    """CWRU (D118, D125): class = fault location, outer race with its clock hour ('outer@6');
    group = the faulty bearing's ID. Normal files (no fault) have neither: their healthy
    bearings are background, not test bearings (D125)."""
    labels, groups = [], []
    for c in captures.itertuples():
        if c.fault_location == "none":
            labels.append(None)
            groups.append(None)
            continue
        labels.append(f"outer@{int(c.clock_hour)}" if c.fault_location == "outer" else c.fault_location)
        groups.append(c.de_bearing if c.fault_end == "DE" else c.fe_bearing)
    return labels, groups


# Herning operating point and kappa as the MSSP code (D55), the same values as configs/band_check_herning.yaml.
# Viscosities: the MSSP fallback (the file's lubricant metadata has none); d_pw: the bearing's pitch diameter.
HERNING = {"rpm_per_cmd_hz": 59.5, "rpm_min": 50, "rpm_max": 3600, "temp_min_c": 38,
           "not_rotating_rpm_target": 100, "cooldown_temp_target": 25,
           "d_pw_mm": 38.0, "nu_40_cst": 22.0, "nu_100_cst": 4.1}


def herning(captures):
    """Herning (D55): label = kappa of the sweep's operating point (a number; regression, D58);
    group = None (random split, D168).

    rpm = frequency-converter command x 59.5 (the measured rpm is unreliable); temperature = the
    Omron process value. Sweeps excluded by D55 get no label: bearing not rotating (rpm target 100),
    cool-down tail (temperature target 25), rpm outside 50-3600 (this includes rpm target 0, VFD off),
    temperature below 38 C. Needs the telemetry columns, which windows.select joins onto the captures.
    """
    h = HERNING
    labels = []
    for c in captures.itertuples():
        rpm = c.telem_vfd_cmd_hz * h["rpm_per_cmd_hz"]
        temp = c.telem_omron_pv_c
        excluded = (c.telem_rpm_target == h["not_rotating_rpm_target"] or c.telem_temp_target == h["cooldown_temp_target"]
                    or not h["rpm_min"] <= rpm <= h["rpm_max"] or temp < h["temp_min_c"])
        labels.append(None if excluded else kappa.kappa(rpm, temp, h["d_pw_mm"], h["nu_40_cst"], h["nu_100_cst"]))
    return labels, [None] * len(labels)


LABELS = {"cwru": cwru, "herning": herning}
