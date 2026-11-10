"""Convert the Paderborn run-to-failure data to the canonical format (D236): one run per experiment.

Usage:
    uv run python scripts/convert_paderborn_rtf.py <folder with the ZIPs, e.g. data/raw/paderborn_rtf> <output root, e.g. data/canonical> [B01 ...]

Reads the Zenodo ZIP archives directly (nothing is unpacked). Without experiment names, every
experiment whose ZIPs are in the folder is converted. Runs that already exist are skipped.
After writing, the HDF5 file is read back and every capture's codes are compared with what was
written (SHA-256), so the archives can be deleted afterwards (D235; done by
scripts/download_paderborn_rtf.py --convert-to, not here).
"""

import hashlib
import os
import sys

import h5py

from harness.data import canonical, paderborn_rtf


def is_converted(root, experiment):
    rid = paderborn_rtf.run_id(experiment)
    return os.path.exists(os.path.join(canonical.run_dir(root, paderborn_rtf.DATASET, rid), "telemetry.parquet"))


def convert_experiment(folder, root, experiment):
    """Convert one experiment and check the written file; returns the conversion notes."""
    rid = paderborn_rtf.run_id(experiment)
    paths = paderborn_rtf.zip_parts(folder, experiment)
    if not paths:
        raise FileNotFoundError(f"no ZIP files for {experiment} in {folder}")
    members = paderborn_rtf.open_members(paths)
    try:
        tables, notes = paderborn_rtf.read_tables(paths, experiment, members)
        print(f"{rid}: {len(tables['captures'])} measurements from {len(paths)} archive(s)", flush=True)
        hashes = {}
        canonical.write_run(root, paderborn_rtf.DATASET, rid, tables,
                            paderborn_rtf.signals(members, tables, hashes))
    finally:
        for z in {z for z, _ in members.values()}:
            z.close()

    # Read back: every capture and channel must hold exactly the codes that were written.
    path = os.path.join(canonical.run_dir(root, paderborn_rtf.DATASET, rid), "signals.h5")
    with h5py.File(path, "r") as f:
        for (capture_id, ch), h in hashes.items():
            if hashlib.sha256(f[f"{capture_id}/{ch}"][:].tobytes()).hexdigest() != h:
                raise ValueError(f"{rid} {capture_id}/{ch}: read-back differs from what was written")
    if len(hashes) != 2 * len(tables["captures"]):
        raise ValueError(f"{rid}: {len(hashes)} signals written, expected {2 * len(tables['captures'])}")
    print(f"{rid}: written and checked" + (f"  notes: {notes}" if notes else ""), flush=True)
    return notes


def convert(folder, root, experiments=None):
    for experiment in experiments or paderborn_rtf.EXPERIMENTS:
        if is_converted(root, experiment):
            print(f"{experiment}: exists, skipped")
        elif paderborn_rtf.zip_parts(folder, experiment):
            convert_experiment(folder, root, experiment)
        elif experiments:
            raise FileNotFoundError(f"no ZIP files for {experiment} in {folder}")
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
