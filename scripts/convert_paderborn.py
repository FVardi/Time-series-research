"""Convert the Paderborn bearing data to the canonical format (D212): one run per bearing and condition.

Usage:
    uv run python scripts/convert_paderborn.py <folder with the unpacked bearing folders, e.g. data/raw/paderborn> <output root, e.g. data/canonical> [bearing ...]

Without bearing codes, all 32 bearings are converted. Runs that already exist are skipped,
so an interrupted conversion can be restarted. Quirks found (D212 item 7) are printed.
"""

import os
import sys

from harness.data import canonical, paderborn


def convert(folder, root, bearings=None):
    missing = [b for b in paderborn.BEARINGS if not os.path.isdir(os.path.join(folder, b))]
    if missing:
        raise FileNotFoundError(f"missing bearing folders in {folder}: {missing}")
    for bearing in bearings or paderborn.BEARINGS:
        for condition in paderborn.CONDITIONS:
            rid = paderborn.run_id(bearing, condition)
            if os.path.exists(os.path.join(canonical.run_dir(root, paderborn.DATASET, rid), "telemetry.parquet")):
                print(f"{rid}: exists, skipped")
                continue
            tables, signals, notes = paderborn.read(folder, bearing, condition)
            canonical.write_run(root, paderborn.DATASET, rid, tables, signals)
            print(f"{rid}" + (f"  notes: {notes}" if notes else ""), flush=True)
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
