"""Integer ADC codes for sources whose values lie on a fixed grid (D6, D20; Paderborn RTF D236, UOS D248).

grid_constants finds the step and offset of the grid from one recording; to_codes turns values into
integer codes on that grid, physical value = code * step + offset, and stops if a value is off the
grid, a code does not fit the integer type, or the reconstruction is further from the source than
max_ulp units in the last place of the recording's largest absolute value.
"""

import numpy as np

GRID_TOLERANCE_STEPS = 1e-6  # a value further than this from the grid stops the conversion


def grid_constants(x):
    """Step and offset of the ADC grid that the values x lie on.

    The step is estimated from the smallest difference between distinct values (ignoring differences
    far below the median one) and refined by a
    least-squares fit of value against grid index; the offset is taken as the grid point nearest
    zero, so that codes are centred on 0 (exactly 0 is tried when it is within float noise). Among the
    fitted constants and their roundings to 4-12 significant digits (instruments often store round decimals, e.g. 0.000323335 V), the pair whose
    reconstruction matches x exactly most often is returned.
    """
    u = np.unique(x)
    d = np.diff(u)
    # First guess of the step: the smallest gap between distinct values, ignoring gaps far below the
    # typical one (a stray value such as -6.4e-302 next to 0 in one UOS file, D248 note)
    k = np.rint((u - u[0]) / np.min(d[d >= 0.5 * np.median(d)]))
    c0, step = np.linalg.lstsq(np.vstack([np.ones_like(k), k]).T, u, rcond=None)[0]
    offset = c0 - np.rint(c0 / step) * step
    candidates = [(step, offset)] + [(float(f"{step:.{p}g}"), float(f"{offset:.{p}g}")) for p in range(4, 13)]
    if abs(offset) < 1e-6 * step:  # an offset of zero up to float noise (UOS): also try exactly 0
        candidates += [(s, 0.0) for s, _ in candidates]
        # With offset 0, a value with a small code n is n * step; dividing it by n recovers the
        # instrument's step from the data itself (exactly for n = 1, 2, 4), independent of the
        # platform's linear algebra (UOS, D248 notes). Some recordings lack code 1, so the few
        # smallest values of both signs are used.
        small = np.concatenate([u[u > 0][:5], u[u < 0][-5:]])
        for v in small:
            n = np.rint(v / step)
            if 1 <= abs(n) <= 16 and abs(v / n - step) < 1e-6 * step:
                candidates.append((float(v / n), 0.0))
    return max(candidates, key=lambda c: int(np.sum(np.rint((x - c[1]) / c[0]) * c[0] + c[1] == x)))


def to_codes(x, step, offset, where, dtype, max_ulp):
    """Integer codes (dtype) of x on the given grid; stops if x is off the grid or not reconstructible."""
    k = (x - offset) / step
    codes = np.rint(k)
    if np.max(np.abs(k - codes)) > GRID_TOLERANCE_STEPS:
        raise ValueError(f"{where}: values not on the ADC grid (step {step!r}, offset {offset!r})")
    info = np.iinfo(dtype)
    if codes.min() < info.min or codes.max() > info.max:
        raise ValueError(f"{where}: codes outside {np.dtype(dtype).name}")
    rebuilt = codes * step + offset
    # float64 resolution of the recording's values: one ulp of its largest magnitude
    if np.max(np.abs(rebuilt - x)) > max_ulp * np.spacing(np.max(np.abs(x))):
        raise ValueError(f"{where}: reconstruction differs from the source by more than {max_ulp} ulp")
    return codes.astype(dtype)
