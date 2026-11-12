"""D266: one probe row per capture, the mean of its windows' representations (XJTU-SY minutes)."""
import numpy as np
import pandas as pd

from harness import evaluate


def test_mean_per_capture_follows_first_appearance():
    # two bearings; capture m1 of B1, m2 of B1 and m1 of B2 with three windows each, rows in Z order
    table = pd.DataFrame({"run_id": ["B1"] * 6 + ["B2"] * 3, "capture_id": ["m1"] * 3 + ["m2"] * 3 + ["m1"] * 3})
    codes, first = evaluate.capture_codes(table)
    assert codes.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]  # same capture id in another run is another capture
    assert first.tolist() == [0, 3, 6]
    Z = np.arange(18, dtype=float).reshape(9, 2)
    pooled = evaluate.mean_per_capture(Z, codes)
    assert np.allclose(pooled, [Z[0:3].mean(0), Z[3:6].mean(0), Z[6:9].mean(0)])
