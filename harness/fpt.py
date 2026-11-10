"""First prediction time (FPT) of a run-to-failure recording (D191, D229, D230).

Li, Lei, Lin, Ding (2015), "An improved exponential model for predicting remaining useful life of
rolling element bearings", IEEE Trans. Ind. Electron. 62(12), 7762-7773, Section III:
  1. kurtosis of each sample is the monitoring index;
  2. a two-sided 3-sigma interval [mu - 3 sd, mu + 3 sd] from a normal-operation reference segment;
  3. trigger: FPT_0 = first value outside the interval; FPT_l = first start of l + 1 consecutive
     values outside it, l = 1, 2, ...; stop when FPT_l = FPT_(l-1). No such run: no FPT ("N/A").
Our choices (D230 item 4, not in the source): the reference is the first 20 samples (minutes), the
search starts after it, and sd is the sample standard deviation (ddof = 1).
"""

import numpy as np

REFERENCE = 20  # minutes (D230 item 4)

# XJTU-SY FPT per bearing (minute, 1-based), horizontal channel (D229, D230), as computed on
# 2026-11-05 from the converted data; tests/test_xjtu_sy_tasks.py recomputes it.
# Bearings left out of the health-stage and health-indicator tasks (D230 item 5) are not listed.
XJTU_SY = {"Bearing1_1": 76, "Bearing1_2": 44, "Bearing1_3": 60, "Bearing1_4": 80, "Bearing1_5": 36,
           "Bearing2_1": 455, "Bearing2_2": 48, "Bearing2_3": 128, "Bearing2_5": 142,
           "Bearing3_1": 2410, "Bearing3_3": 340}
XJTU_SY_LEFT_OUT = ("Bearing2_4", "Bearing3_2", "Bearing3_4", "Bearing3_5")  # D230 item 5


def kurtosis(x):
    """Kurtosis of one sample: fourth central moment over the squared variance (not excess; about 3
    for a Gaussian signal, as in Li et al.'s Figs. 5 and 12)."""
    xc = x - x.mean()
    return np.mean(xc**4) / np.mean(xc**2) ** 2


def _first_run(outside, start, k):
    """First index t >= start where outside[t:t+k] are all True, else None."""
    for t in range(start, len(outside) - k + 1):
        if outside[t:t + k].all():
            return t
    return None


def li2015(index, reference=REFERENCE):
    """FPT (1-based position) of the monitoring index (one value per sample, in time order), or None."""
    index = np.asarray(index, dtype=np.float64)
    mu, sd = index[:reference].mean(), index[:reference].std(ddof=1)
    outside = np.abs(index - mu) > 3 * sd
    previous = _first_run(outside, reference, 1)  # FPT_0
    if previous is None:
        return None
    l = 1
    while True:
        current = _first_run(outside, reference, l + 1)  # FPT_l
        if current is None:
            return None
        if current == previous:
            return current + 1
        previous, l = current, l + 1
