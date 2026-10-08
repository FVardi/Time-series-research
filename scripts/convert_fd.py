"""Convert FD-A or FD-B (TF-C's version) to the canonical format (D39).

Usage (one call per folder):
    uv run python scripts/convert_fd.py data/raw/fd_a fd_a data/canonical
    uv run python scripts/convert_fd.py data/raw/fd_b fd_b data/canonical
"""

import sys

from harness.data import canonical, fd


def convert(folder, run_id, root):
    tables, signals = fd.read(folder, run_id)
    print(f"Converting {folder} -> {canonical.run_dir(root, fd.DATASET, run_id)}")
    canonical.write_run(root, fd.DATASET, run_id, tables, signals)
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
