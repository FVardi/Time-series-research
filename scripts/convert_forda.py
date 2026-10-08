"""Convert FordA to the canonical format.

Usage:
    uv run python scripts/convert_forda.py <folder with FordA_TRAIN/TEST files> <output root, e.g. data/canonical>
"""

import sys

from harness.data import canonical, forda


def convert(folder, root):
    tables, signals = forda.read(folder)
    print(f"Converting {folder} -> {canonical.run_dir(root, forda.DATASET, forda.RUN_ID)}")
    canonical.write_run(root, forda.DATASET, forda.RUN_ID, tables, signals)
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
