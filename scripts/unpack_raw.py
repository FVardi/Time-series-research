"""Unpack the dataset archives downloaded on 2026-11-18 into data/raw, then delete each archive.

Usage (from the repo root, on the Windows machine):
    uv run python scripts/unpack_raw.py

Each archive is extracted in full; every extracted file is checked against the size stored in the
archive, and the archive is deleted only if all files match. Files that already exist with the
right size (from an earlier, interrupted run) are skipped, so the script can simply be run again.
Archives that are not found are reported and skipped.
"""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

RAW = Path("data/raw")
DOWNLOADS = Path.home() / "Downloads"
DCASE = RAW / "dcase2021_task2"
DCASE_TYPES = ["fan", "gearbox", "pump", "slider", "ToyCar", "ToyTrain", "valve"]


def jobs():
    """(archive, destination folder) pairs, in the order they are processed."""
    out = [
        (RAW / "ucr_archive" / "Univariate2018_ts.zip", RAW / "ucr_archive"),
        (RAW / "uea_archive" / "Multivariate2018_ts.zip", RAW / "uea_archive"),
    ]
    # TSER: one zip per dataset, unpacked into a folder of the same name.
    for z in sorted((RAW / "tser").glob("*.zip")):
        out.append((z, z.with_suffix("")))
    # AdaTime: one zip per dataset, unpacked into the folder it was saved in.
    for z in sorted(RAW.glob("adatime_*/*.zip")):
        out.append((z, z.parent))
    out += [
        (DOWNLOADS / "ptb-xl-1.0.3.zip", RAW / "ptbxl"),
        (DOWNLOADS / "speech_commands_v0.02.tar.gz", RAW / "speech_commands_v2"),
        (DOWNLOADS / "ground_truth_data.zip", DCASE),
    ]
    # DCASE 2021 Task 2: development set -> dev_data/, additional training + evaluation set -> eval_data/
    # (the folder layout of the official baseline).
    for t in DCASE_TYPES:
        out.append((DOWNLOADS / f"dev_data_{t}.zip", DCASE / "dev_data"))
    for t in DCASE_TYPES:
        out.append((DOWNLOADS / f"eval_data_{t}_train.zip", DCASE / "eval_data"))
        out.append((DOWNLOADS / f"eval_data_{t}_test.zip", DCASE / "eval_data"))
    return out


def files_in(archive):
    """Yield (name, size, file object) for each regular file, reading the archive front to back."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if not m.is_dir():
                    with z.open(m) as f:
                        yield m.filename, m.file_size, f
    else:  # .tar.gz: iterate in order, so the gzip stream is never re-read from the start
        with tarfile.open(archive, "r:gz") as t:
            for m in t:
                if m.isfile():
                    with t.extractfile(m) as f:
                        yield m.name, m.size, f


def unpack(archive, dest):
    """Extract every file not yet present with the right size; return True if all files check out."""
    dest = dest.resolve()
    sizes = {}
    for name, size, src in files_in(archive):
        out = (dest / name).resolve()
        if dest not in out.parents:  # refuse paths that would land outside dest
            raise ValueError(f"unsafe path in {archive}: {name}")
        sizes[out] = size
        if out.exists() and out.stat().st_size == size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    bad = [p for p, s in sizes.items() if p.stat().st_size != s]
    if bad:
        print(f"  size mismatch for {len(bad)} files, e.g. {bad[0]}; archive kept")
    return not bad


if __name__ == "__main__":
    for archive, dest in jobs():
        if not archive.exists():
            print(f"skip (not found): {archive}")
            continue
        print(f"{archive} -> {dest}")
        if unpack(archive, dest):
            os.remove(archive)
            print(f"  ok ({archive.name} deleted)")
