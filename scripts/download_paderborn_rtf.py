"""Download the Paderborn run-to-failure bearing data (Zenodo record 10868257), optionally
converting each experiment to the canonical format and deleting its archives (D235).

Data set: "Run-to-failure data set of ball bearings subjected to time-varying load and speed
conditions", LDM, Paderborn University (Aimiyekagbon, 2024), CC BY 4.0.
Cite: Javanmardi et al. (2024), PHM Society European Conference 8(1),
doi 10.36001/phme.2024.v8i1.4101.

Download only: every file of the record (48 ZIP parts, about 152 GB, plus the description PDF)
goes unchanged into --out.
    python scripts/download_paderborn_rtf.py --out data/raw/paderborn_rtf [--only B01.zip ...]

Download and convert (D235): one experiment at a time, its ZIP parts are downloaded, converted
to <root>/paderborn_rtf/<experiment>/ (scripts/convert_paderborn_rtf.py, which checks the written
file), its small files (log PDF, CSVs) are unpacked into --out/<experiment>/, and only then are
its ZIP parts deleted. The description PDF is kept. Rerunning skips finished experiments.
    uv run python scripts/download_paderborn_rtf.py --out data/raw/paderborn_rtf --convert-to data/canonical [--experiments B01 ...]

- File names, sizes and MD5 checksums come from the Zenodo records API.
- Interrupted downloads resume from the `.part` file; dropped connections are retried automatically.
- Each new download is checked against its MD5 before it gets its final name.
- Files already in --out with the right size are skipped (use --verify to re-check their MD5).
- Stops before downloading if the disk does not have room.
"""

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "10868257"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK = 8 * 1024 * 1024  # 8 MB per read


def list_files():
    """Return [(name, size_in_bytes, md5_hex, url)] for every file of the record."""
    with urllib.request.urlopen(API_URL, timeout=60) as response:
        record = json.load(response)
    files = []
    for entry in record["files"]:
        algorithm, checksum = entry["checksum"].split(":")
        assert algorithm == "md5", entry["checksum"]
        url = f"https://zenodo.org/records/{RECORD_ID}/files/{entry['key']}?download=1"
        files.append((entry["key"], entry["size"], checksum, url))
    return files


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, size, md5, attempts=20, wait_s=30):
    """Download url to target via target.part, resuming if the .part file exists.

    A dropped or stalled connection is retried (up to `attempts` times, `wait_s` apart),
    each time resuming from the bytes already in the .part file.
    """
    part = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        done = part.stat().st_size if part.exists() else 0
        if done >= size:
            break
        request = urllib.request.Request(url)
        if done:
            request.add_header("Range", f"bytes={done}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if done and response.status != 206:  # server ignored the range: start over
                    done = 0
                with open(part, "ab" if done else "wb") as f:
                    while block := response.read(CHUNK):
                        f.write(block)
                        done += len(block)
                        print(f"\r  {done / 1e9:6.2f} / {size / 1e9:.2f} GB", end="", flush=True)
        except (OSError, http.client.HTTPException) as e:  # includes URLError, timeouts, resets
            print(f"\n  connection problem ({e!r}); retry {attempt}/{attempts} in {wait_s} s", flush=True)
            time.sleep(wait_s)
            continue
        print()
    if not part.exists() or part.stat().st_size != size or md5_of(part) != md5:
        sys.exit(f"Size or MD5 mismatch for {target.name}; delete {part} and run again.")
    part.rename(target)


def missing(out, files, verify):
    """The files not yet complete in out, and the bytes still needed for them."""
    todo, needed = [], 0
    for name, size, md5, url in files:
        target = out / name
        if target.exists() and target.stat().st_size == size:
            if verify and md5_of(target) != md5:
                sys.exit(f"MD5 mismatch for existing {name}; delete it and run again.")
            continue
        part = out / (name + ".part")
        todo.append((name, size, md5, url))
        needed += size - (part.stat().st_size if part.exists() else 0)
    return todo, needed


def download_all(out, files, verify, extra_bytes=0):
    """Download the missing files; extra_bytes = room to keep free for the conversion output."""
    todo, needed = missing(out, files, verify)
    free = shutil.disk_usage(out).free
    print(f"{len(files)} files in selection, {len(todo)} to download, {needed / 1e9:.1f} GB needed"
          + (f" (+{extra_bytes / 1e9:.1f} GB for the conversion)" if extra_bytes else "")
          + f", {free / 1e9:.1f} GB free.")
    if needed + extra_bytes > free:
        sys.exit("Not enough disk space.")
    for i, (name, size, md5, url) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name}")
        download(url, out / name, size, md5)


def unpack_small_files(out, experiment, zip_paths):
    """Unpack everything except vibrationData/ (the log PDF and CSVs) into out/<experiment>/.

    Multi-part archives have the top folder <experiment>_part<k>/; the files go to
    out/<experiment>/ for every experiment alike (D235).
    """
    for p in zip_paths:
        with zipfile.ZipFile(p) as z:
            for info in z.infolist():
                if "/vibrationData/" in info.filename or info.is_dir():
                    continue
                target = out / experiment / info.filename.split("/", 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if target.stat().st_size != info.file_size:
                    sys.exit(f"{info.filename}: unpacked size differs from the archive")


def convert_all(out, root, files, experiments, verify):
    import convert_paderborn_rtf as conv  # needs the project environment (uv run)

    by_experiment = {}
    for f in files:
        m = re.fullmatch(r"(B\d\d)(_part\d+)?\.zip", f[0])
        if m:
            by_experiment.setdefault(m.group(1), []).append(f)
    docs = [f for f in files if not re.fullmatch(r"B\d\d(_part\d+)?\.zip", f[0])]
    download_all(out, docs, verify)  # the description PDF, kept as downloaded

    for experiment in experiments or sorted(by_experiment):
        parts = by_experiment[experiment]
        zips = [out / name for name, *_ in parts]
        if conv.is_converted(root, experiment):
            if any(z.exists() for z in zips):  # converted earlier, archives not yet cleaned up
                unpack_small_files(out, experiment, [z for z in zips if z.exists()])
                for z in zips:
                    z.unlink(missing_ok=True)
            print(f"{experiment}: converted, skipped")
            continue
        print(f"=== {experiment}")
        # Room for the archives plus the converted run (smaller than the archives).
        download_all(out, parts, verify, extra_bytes=sum(size for _, size, _, _ in parts))
        conv.convert_experiment(str(out), root, experiment)  # stops on any problem; archives kept
        unpack_small_files(out, experiment, zips)
        for z in zips:
            z.unlink()
        print(f"{experiment}: archives deleted")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="raw data folder, e.g. data/raw/paderborn_rtf")
    parser.add_argument("--only", nargs="+", help="download only: these file names")
    parser.add_argument("--convert-to", help="canonical root, e.g. data/canonical: convert and delete archives (D235)")
    parser.add_argument("--experiments", nargs="+", help="with --convert-to: only these experiments, e.g. B01")
    parser.add_argument("--verify", action="store_true", help="also check MD5 of files already present")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_files()

    if args.convert_to:
        if args.only:
            sys.exit("--only is for download only; use --experiments with --convert-to.")
        convert_all(out, args.convert_to, files, args.experiments, args.verify)
    else:
        if args.only:
            unknown = set(args.only) - {name for name, *_ in files}
            if unknown:
                sys.exit(f"Not in the record: {sorted(unknown)}")
            files = [f for f in files if f[0] in args.only]
        download_all(out, files, args.verify)
    print("Done.")


if __name__ == "__main__":
    main()
