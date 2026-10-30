"""Compute and store a feature set for every capture of a dataset (D172, D181).

Usage:
    uv run python scripts/derive_features.py <canonical root> <dataset> <feature set> <output file>
    e.g. uv run python scripts/derive_features.py data/canonical herning band data/derived/herning/band_features.h5
         uv run python scripts/derive_features.py data/canonical herning mssp data/derived/herning/mssp_stats.h5

Feature sets (harness/windows.py, FEATURE_SETS): band = per-window band features (D165), mssp = MSSP
statistics per band over the whole capture (D181). Reads the canonical signals (read-only, D5)
capture by capture and writes the features, with the settings used, into one HDF5 file
(harness/stored_features.py). Configs read it with windows: features_file. Every capture is stored,
labelled or not, so the split and labels can change without recomputing. Rerun after any change to
the feature definitions (loading stops if the settings differ).
"""

import os
import sys

from harness import stored_features
from harness.data import canonical
from harness.windows import FEATURE_SETS


def main(root, dataset, feature_set, path):
    module = FEATURE_SETS[feature_set]
    meta = canonical.load_metadata(root, dataset)
    channels = list(meta["channels"].channel.unique())
    fs = meta["channels"].sample_rate_hz.unique()
    if len(fs) != 1 or meta["captures"].n_samples.nunique() != 1:
        raise ValueError("all channels need one sample rate and all captures one length")
    caps = meta["captures"]
    first = canonical.load_captures(root, dataset, caps.run_id[0], [caps.capture_id[0]], channels)
    row_shape = module.compute(first, channels, fs[0]).shape[1:]  # (windows, feature channels)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"{feature_set}: {len(caps)} captures x {row_shape} from {channels} -> {path}", flush=True)
    with stored_features.create(path, len(caps), row_shape, module.settings(channels, fs[0]),
                                module.names(channels)) as f:
        for i, c in enumerate(caps.itertuples()):
            x = canonical.load_captures(root, dataset, c.run_id, [c.capture_id], channels)  # (1, samples, channels)
            f["features"][i] = module.compute(x, channels, fs[0])[0]
            f["run_id"][i], f["capture_id"][i] = c.run_id, c.capture_id
            if i % 250 == 0:
                print(f"  {i} / {len(caps)}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main(*sys.argv[1:5])
