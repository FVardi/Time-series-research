"""Convert the CWRU bearing data to the canonical format (D139): one run per source .mat file.

Usage:
    uv run python scripts/convert_cwru.py <folder with the .mat files, e.g. data/raw/cwru> <output root, e.g. data/canonical>

Stops if a file of the table is missing, so a partial conversion cannot pass unnoticed.
"""

import os
import sys

from harness.data import canonical, cwru


def convert(folder, root, files=None):
    """Convert every file of cwru.file_table() (or only the file numbers in 'files')."""
    table = [e for e in cwru.file_table() if files is None or e["file"] in files]
    missing = [e["file"] for e in table if not os.path.exists(os.path.join(folder, f"{e['file']}.mat"))]
    if missing:
        raise FileNotFoundError(f"missing source files in {folder}: {missing}")
    for entry in table:
        tables, signals = cwru.read(folder, entry)
        rid = cwru.run_id(entry["file"])
        canonical.write_run(root, cwru.DATASET, rid, tables, signals)
        ignored = cwru.IGNORED.get(entry["file"])
        print(f"{entry['file']}.mat -> {rid}" + (f"  (ignored variables: {sorted(ignored)})" if ignored else ""))
    print(f"Done: {len(table)} runs.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
