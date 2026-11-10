"""Convert the DCASE 2022 Task 2 development data, machine type bearing, to the canonical format (D251):
one run per section.

Usage:
    uv run python scripts/convert_dcase2022_bearing.py <folder with train/, test/ and the attribute CSVs, e.g. data/raw/dcase2022_bearing/dev_bearing/bearing> <output root, e.g. data/canonical> [section ...]

Without sections (00, 01, 02), all three are converted. Runs that already exist are skipped.
"""

import os
import sys

from harness.data import canonical, dcase2022_bearing as dcase


def convert(folder, root, sections=None):
    for section in sections or dcase.SECTIONS:
        rid = dcase.run_id(section)
        if os.path.exists(os.path.join(canonical.run_dir(root, dcase.DATASET, rid), "telemetry.parquet")):
            print(f"{rid}: exists, skipped")
            continue
        tables, signals, notes = dcase.read(folder, section)
        canonical.write_run(root, dcase.DATASET, rid, tables, signals)
        print(f"{rid}: {len(tables['captures'])} clips  notes: {notes}", flush=True)
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
