"""Inputs cut into windows, for datasets whose split is set in the config (D147).

Used by scripts/pretrain.py when the config has a 'split' section. Steps:
  select   - captures matching filters on the captures table (and the stream's sample rate)
  cut      - load, resample to one rate if needed (D136), cut consecutive windows (D137)
  standardisation / standardise - one mean and SD per channel from given windows (D137)
  check_separate - no capture and no group (bearing) of the test set in the pretraining data (D33, D121)
prepare() runs these for the 'train', 'test' and 'pretrain_only' parts of a config.
"""

from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from harness.data import canonical
from harness.labels import LABELS

PARTS = ("train", "test", "pretrain_only")


def stream_rates(meta):
    """Sample rate of each (run, stream). All channels of a stream share one rate (D1)."""
    rates = meta["channels"].groupby(["run_id", "stream"]).sample_rate_hz.unique()
    if (rates.str.len() != 1).any():
        raise ValueError("a stream has channels with different sample rates")
    return rates.str[0].rename("sample_rate_hz").reset_index()


def select(meta, filters):
    """Captures whose columns match every filter {column: allowed value(s)}.

    The stream's sample rate is joined onto each capture, so it can be filtered on too.
    """
    caps = meta["captures"].merge(stream_rates(meta), on=["run_id", "stream"])
    keep = np.ones(len(caps), dtype=bool)
    for column, allowed in filters.items():
        allowed = allowed if isinstance(allowed, list) else [allowed]
        keep &= caps[column].isin(allowed).to_numpy()
    return caps[keep].reset_index(drop=True)


def cut(root, dataset, captures, channels, length, rate_hz):
    """Load each capture, resample it to rate_hz if it has another rate (D136), and cut
    consecutive windows of 'length' samples from its first sample; the incomplete last
    window is dropped (D137).

    Returns X (n_windows, length, n_channels) and a table with one row per window:
    run_id, capture_id, start and stop (samples of the signal at rate_hz, stop exclusive)
    and the capture's own rate (resampled_from_hz, equal to rate_hz if not resampled).
    """
    X, rows = [], []
    for c in captures.itertuples():
        x = canonical.load_captures(root, dataset, c.run_id, [c.capture_id], channels)[0]  # (samples, channels)
        if c.sample_rate_hz != rate_hz:
            # polyphase resampling with scipy's built-in anti-alias filter, e.g. 48 -> 12 kHz is up 1, down 4
            ratio = Fraction(int(rate_hz), int(c.sample_rate_hz))
            x = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)
        n = len(x) // length
        X.append(x[: n * length].reshape(n, length, len(channels)))
        rows += [{"run_id": c.run_id, "capture_id": c.capture_id, "start": i * length, "stop": (i + 1) * length,
                  "resampled_from_hz": c.sample_rate_hz} for i in range(n)]
    return np.concatenate(X), pd.DataFrame(rows)


def standardisation(X):
    """One mean and one standard deviation per channel, over all samples of the windows X (D137)."""
    return {"mean": X.mean(axis=(0, 1)), "std": X.std(axis=(0, 1))}


def standardise(X, stats):
    """The same fixed change of units for every window: (x - mean) / std per channel (D137)."""
    return (X - stats["mean"]) / stats["std"]


def check_separate(table):
    """Stop if a test capture or a test group (e.g. bearing) is also in the pretraining data
    ('train' and 'pretrain_only' parts) (D33, D121)."""
    test = table[table.split == "test"]
    pre = table[table.split.isin(["train", "pretrain_only"])]
    shared = set(zip(test.run_id, test.capture_id)) & set(zip(pre.run_id, pre.capture_id))
    if shared:
        raise ValueError(f"{len(shared)} captures are in both pretraining and test, e.g. {sorted(shared)[:5]}")
    shared = set(test.group.dropna()) & set(pre.group.dropna())
    if shared:
        raise ValueError(f"groups in both pretraining and test: {sorted(shared)}")


def prepare(cfg, meta):
    """Windows of the 'train', 'test' and 'pretrain_only' parts of cfg['split'].

    Returns {part: (X, table)}; each table has split, row (index in X), run_id, capture_id,
    channels, start, stop, resampled_from_hz, label and group. 'train' and 'test' must have a
    class for every window; 'pretrain_only' is used without labels. Parts may not share captures.
    """
    w = cfg["windows"]
    out = {}
    for part in PARTS:
        caps = select(meta, cfg["split"][part])
        caps["label"], caps["group"] = LABELS[cfg["labels"]](caps)
        if part != "pretrain_only" and caps.label.isna().any():
            raise ValueError(f"'{part}' contains captures without a class: {caps[caps.label.isna()].run_id.tolist()[:5]}")
        X, table = cut(cfg["data_root"], cfg["dataset"], caps, cfg["channels"], w["length"], w["sample_rate_hz"])
        # class and group of each window's capture (a lookup keeps the rows in the order of X)
        key = list(zip(caps.run_id, caps.capture_id))
        label, group = dict(zip(key, caps.label)), dict(zip(key, caps.group))
        rows = list(zip(table.run_id, table.capture_id))
        table["label"] = [label[k] for k in rows]
        table["group"] = [group[k] for k in rows]
        table.insert(0, "split", part)
        table.insert(1, "row", np.arange(len(table)))
        table.insert(4, "channels", ";".join(cfg["channels"]))
        out[part] = (X, table)
    used = [set(zip(t.run_id, t.capture_id)) for _, t in out.values()]
    if used[0] & used[1] or used[0] & used[2] or used[1] & used[2]:
        raise ValueError("the parts of the split share captures")
    return out
