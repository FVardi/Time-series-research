"""Convert the HUST bearing data to the canonical format (D223): one run per file.

Usage:
    uv run python scripts/convert_hust.py <folder with the 99 .mat files> <output root, e.g. data/canonical> [file ...]

The folder as downloaded is data/raw/hust/HUST bearing a practical dataset for ball bearing fault diagnosis/HUST bearing dataset.
Without file names (e.g. I402), all 99 files are converted. Runs that already exist are skipped,
so an interrupted conversion can be restarted. Quirks found are printed.
"""

import os
import sys

from harness.data import canonical, hust


def convert(folder, root, files=None):
    missing = [n for n in hust.FILES if not os.path.exists(os.path.join(folder, f"{n}.mat"))]
    if missing:
        raise FileNotFoundError(f"missing files in {folder}: {missing}")
    for name in files or hust.FILES:
        rid = hust.run_id(name)
        if os.path.exists(os.path.join(canonical.run_dir(root, hust.DATASET, rid), "telemetry.parquet")):
            print(f"{rid}: exists, skipped")
            continue
        tables, signals, notes = hust.read(folder, name)
        canonical.write_run(root, hust.DATASET, rid, tables, signals)
        print(f"{rid}" + (f"  notes: {notes}" if notes else ""), flush=True)
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
