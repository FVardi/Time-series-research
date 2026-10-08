"""Convert one Herning source file to the canonical format.

Usage:
    uv run python scripts/convert_herning.py <source.h5> <output root, e.g. data/canonical>

Reads the source file only; writes <output root>/herning/<run_id>/.
"""

import sys

import h5py

from harness.data import canonical, herning


def signals(path, tables):
    """Yield (capture_id, channel, codes, scale, offset, units) for every sweep and channel."""
    channels = tables["channels"].set_index("channel")
    with h5py.File(path, "r") as f:
        for n, row in enumerate(tables["captures"].itertuples()):
            sweep = f["sweeps"][row.source_name]
            for ch in herning.CHANNELS:
                c = channels.loc[ch]
                # The channels table holds one scale/offset/rate per channel, taken from the
                # first sweep. Stop if any sweep differs, instead of storing a wrong value.
                a = sweep[ch].attrs
                if (a["y_increment"], a["y_origin"], a["sample_rate"], a["x_origin"]) != (
                        c.scale, c.offset, c.sample_rate_hz, c.x_origin_s):
                    raise ValueError(f"{row.source_name}/{ch}: scope settings differ from the first sweep")
                codes = herning.read_codes(f, row.source_name, ch, c.scale, c.offset)
                yield row.capture_id, ch, codes, c.scale, c.offset, c.units
            if n % 250 == 0:
                print(f"  {n} / {len(tables['captures'])} sweeps", flush=True)


def convert(path, root, max_captures=None):
    tables = herning.read_tables(path, max_captures)
    run_id = tables["runs"].run_id[0]
    print(f"Converting {path} -> {canonical.run_dir(root, herning.DATASET, run_id)}")
    canonical.write_run(root, herning.DATASET, run_id, tables, signals(path, tables))
    print("Done.")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
