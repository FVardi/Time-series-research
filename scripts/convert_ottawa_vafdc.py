"""Convert the University of Ottawa UODS-VAFDC data to the canonical format (D249): one run per bearing.

Usage:
    uv run python scripts/convert_ottawa_vafdc.py "<dataset folder>" <output root, e.g. data/canonical> [bearing ...]

The dataset folder is the one holding 1_Healthy/ ... 5_Cage_Faults/, e.g.
data/raw/ottawa_vafdc/University of Ottawa Ball-bearing Vibration and Acoustic Fault Data under Constant Load and Speed Conditions (UODS-VAFDC)
Without bearing numbers (1-20), all 20 bearings are converted. Runs that already exist are skipped,
so an interrupted conversion can be restarted. Quirks found are printed.
"""

import os
import sys

from harness.data import canonical, ottawa_vafdc


def convert(folder, root, bearings=None):
    missing = [n for b in ottawa_vafdc.BEARINGS for n in ottawa_vafdc.files_of(b)
               if not os.path.exists(ottawa_vafdc.path_of(folder, n))]
    if missing:
        raise FileNotFoundError(f"missing files in {folder}: {missing}")
    for bearing in bearings or ottawa_vafdc.BEARINGS:
        rid = ottawa_vafdc.run_id(bearing)
        if os.path.exists(os.path.join(canonical.run_dir(root, ottawa_vafdc.DATASET, rid), "telemetry.parquet")):
            print(f"{rid}: exists, skipped")
            continue
        tables, signals, notes = ottawa_vafdc.read(folder, bearing)
        canonical.write_run(root, ottawa_vafdc.DATASET, rid, tables, signals)
        print(f"{rid}" + (f"  notes: {notes}" if notes else ""), flush=True)
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], [int(b) for b in sys.argv[3:]] or None)
