"""Convert the UOS multi-domain vibration data to the canonical format (D248): one run per file.

Usage:
    uv run python scripts/convert_uos.py <folder holding the three subset folders, e.g. data/raw/uos> <output root, e.g. data/canonical> [file ...]

Without file names (e.g. H_IR_8_6204_1000), all 1,152 files are converted. Runs that already exist
are skipped, so an interrupted conversion can be restarted. Quirks found are printed.
"""

import os
import sys

from harness.data import canonical, uos


def convert(folder, root, files=None):
    found = uos.find_files(folder)
    expected = uos.expected_files()
    missing, extra = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(f"files missing: {missing[:10]} ({len(missing)}); unexpected: {extra[:10]} ({len(extra)})")
    todo = files or expected
    for i, name in enumerate(todo, 1):
        rid = uos.run_id(name)
        if os.path.exists(os.path.join(canonical.run_dir(root, uos.DATASET, rid), "telemetry.parquet")):
            continue  # exists: skipped
        tables, signals, notes = uos.read(folder, found[name], name)
        canonical.write_run(root, uos.DATASET, rid, tables, signals)
        print(f"[{i}/{len(todo)}] {rid}" + (f"  notes: {notes}" if notes else ""), flush=True)
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
