"""Convert the XJTU-SY bearing data to the canonical format (D224): one run per bearing.

Usage:
    uv run python scripts/convert_xjtu_sy.py <dataset folder, e.g. data/raw/xjtu_sy/XJTU-SY_Bearing_Datasets> <output root, e.g. data/canonical> [bearing ...]

Without bearing names (e.g. Bearing1_1), all 15 bearings are converted. Runs that already exist
are skipped, so an interrupted conversion can be restarted. Quirks found are printed.
"""

import os
import sys

from harness.data import canonical, xjtu_sy


def convert(folder, root, bearings=None):
    missing = [b for b, (cond, *_) in xjtu_sy.BEARINGS.items() if not os.path.isdir(os.path.join(folder, cond, b))]
    if missing:
        raise FileNotFoundError(f"missing bearing folders in {folder}: {missing}")
    for bearing in bearings or xjtu_sy.BEARINGS:
        rid = xjtu_sy.run_id(bearing)
        if os.path.exists(os.path.join(canonical.run_dir(root, xjtu_sy.DATASET, rid), "telemetry.parquet")):
            print(f"{rid}: exists, skipped")
            continue
        tables, signals, notes = xjtu_sy.read(folder, bearing)
        canonical.write_run(root, xjtu_sy.DATASET, rid, tables, signals)
        print(f"{rid}" + (f"  notes: {notes}" if notes else ""), flush=True)
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
