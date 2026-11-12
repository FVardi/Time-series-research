"""Baseline: the catch24 feature set (catch22 plus mean and standard deviation) (D85, D90).

Computed with pycatch22 1.0.2, the authors' own C implementation (Lubba et al., 2019);
since v1.0 every feature reproduces its definition in hctsa v3.0.0 (D90).
catch22 itself z-normalises each series, so it ignores level and amplitude; catch24 adds
the mean and the standard deviation back (amplitude matters for condition monitoring).
Features per channel, channels one after the other. Nothing is fitted.
Same interface as the other methods (D77).
"""

import numpy as np
import pycatch22


def pretrain(X, s, device):
    """Nothing to fit. The 'model' only carries the settings needed by encode().

    s: the 'catch22' section of the config:
       catch24   - True: add mean and SD (D85)
       mask_nans - False: a feature that cannot be estimated is NaN, and encode() stops (D90);
                   True would silently replace it by a fixed value
    """
    return {"catch24": s["catch24"], "mask_nans": s["mask_nans"]}, []


def save(model, path):
    """Nothing to save (the settings are in the run's config)."""


def encode(model, X):
    """(series, time, channels) -> (series, channels * 24): the features per channel, channels one after the other.

    Column order per channel: pycatch22's order (catch22_all(...)['names']).
    Stops if any feature could not be estimated (NaN), so nothing is filled in silently (D90).
    """
    rows = []
    for series in X:
        row = []
        for c in range(X.shape[2]):
            row += pycatch22.catch22_all(series[:, c], catch24=model["catch24"], mask_nans=model["mask_nans"])["values"]
        rows.append(row)
    Z = np.array(rows, dtype=float)
    if np.isnan(Z).any():
        bad = np.argwhere(np.isnan(Z))
        raise ValueError(f"{len(bad)} catch22 values could not be estimated (NaN), e.g. (series, column) {bad[:5].tolist()}")
    return Z
