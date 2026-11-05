"""Inputs cut into windows, for datasets whose split is set in the config (D147).

Used by scripts/pretrain.py when the config has a 'split' section. Steps:
  select   - captures matching filters on the captures table (and the stream's sample rate)
  random_parts - or: a random split of the labelled captures into train and test (D169)
  split_windows - or: the windows of the labelled captures split at random, stratified by class (D163, D171, D174)
  cut      - load, resample to one rate if needed (D136), cut consecutive windows (D137),
             or overlapping windows with a hop (D226);
             optionally turn each capture's windows into per-window features (D165)
  cut_stored - or: read stored features (band features D172, MSSP statistics D181), one window per capture
  standardisation / standardise - one mean and SD per channel from given windows (D137)
  check_separate - no capture and no group (bearing) of the test set in the pretraining data (D33, D121)
  check_windows_separate - or, for split_windows: no test window in the pretraining data (D33, D176)
prepare() runs these for the 'train', 'test' and 'pretrain_only' parts of a config.
"""

from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split

from harness import band_features, mssp_stats, stored_features, subband_power
from harness.data import canonical
from harness.labels import LABELS

PARTS = ("train", "test", "pretrain_only")
# Feature sets a config can ask for with windows: features (each module has compute, settings, names).
FEATURE_SETS = {"band": band_features,  # per-window band features (D165)
                "mssp": mssp_stats,  # MSSP statistics per band over the whole capture (D181)
                "subbands": subband_power}  # per-window 1/12-octave sub-band power (D186)


def stream_rates(meta):
    """Sample rate of each (run, stream). All channels of a stream share one rate (D1)."""
    rates = meta["channels"].groupby(["run_id", "stream"]).sample_rate_hz.unique()
    if (rates.str.len() != 1).any():
        raise ValueError("a stream has channels with different sample rates")
    return rates.str[0].rename("sample_rate_hz").reset_index()


def select(meta, filters):
    """Captures whose columns match every filter {column: allowed value(s)}.

    The stream's sample rate and the telemetry columns (one row per capture, D2; e.g. Herning's
    operating point) are joined onto each capture, so they can be filtered on and used for labels.
    """
    caps = meta["captures"].merge(stream_rates(meta), on=["run_id", "stream"])
    caps = caps.merge(meta["telemetry"], on=["run_id", "capture_id"], how="left")
    keep = np.ones(len(caps), dtype=bool)
    for column, allowed in filters.items():
        allowed = allowed if isinstance(allowed, list) else [allowed]
        keep &= caps[column].isin(allowed).to_numpy()
    return caps[keep].reset_index(drop=True)


def cut(root, dataset, captures, channels, length, rate_hz, transform=None, hop=None):
    """Load each capture, resample it to rate_hz if it has another rate (D136), and cut
    consecutive windows of 'length' samples from its first sample; the incomplete last
    window is dropped (D137). With hop, a window starts every 'hop' samples, so windows
    overlap when hop < length (D226); without, hop = length (no overlap, D137). If given, transform maps each capture's windows
    (n, length, channels) to what is kept instead, e.g. per-window band features (D165);
    it runs capture by capture, so the raw samples of all captures are never in memory at once.

    Returns X (n_windows, length, n_channels) and a table with one row per window:
    run_id, capture_id, start and stop (samples of the signal at rate_hz, stop exclusive)
    and the capture's own rate (resampled_from_hz, equal to rate_hz if not resampled).
    """
    hop = hop or length
    X, rows = [], []
    for c in captures.itertuples():
        x = canonical.load_captures(root, dataset, c.run_id, [c.capture_id], channels)[0]  # (samples, channels)
        if c.sample_rate_hz != rate_hz:
            # polyphase resampling with scipy's built-in anti-alias filter, e.g. 48 -> 12 kHz is up 1, down 4
            ratio = Fraction(int(rate_hz), int(c.sample_rate_hz))
            x = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)
        n = (len(x) - length) // hop + 1 if len(x) >= length else 0  # equals len(x) // length when hop = length
        idx = np.arange(n)[:, None] * hop + np.arange(length)  # sample indices of each window
        w = x[idx]  # (n, length, channels)
        X.append(transform(w) if transform else w)
        rows += [{"run_id": c.run_id, "capture_id": c.capture_id, "start": i * hop, "stop": i * hop + length,
                  "resampled_from_hz": c.sample_rate_hz} for i in range(n)]
    return np.concatenate(X), pd.DataFrame(rows)


def cut_stored(path, feature_set, captures, channels, length, rate_hz):
    """As cut() with a feature transform, but reading the stored features (D172, D181).

    Only for windows that are whole captures (Herning: one sweep = one window); stops otherwise.
    Returns X (n_captures, feature windows, feature channels) and the same table as cut().
    """
    if (captures.n_samples != length).any() or (captures.sample_rate_hz != rate_hz).any():
        raise ValueError("stored features need windows that are whole captures at the recorded rate")
    expected = FEATURE_SETS[feature_set].settings(channels, rate_hz)  # the file must have been made with these
    X = stored_features.load(path, captures.run_id, captures.capture_id, expected)
    rows = [{"run_id": c.run_id, "capture_id": c.capture_id, "start": 0, "stop": length,
             "resampled_from_hz": c.sample_rate_hz} for c in captures.itertuples()]
    return X, pd.DataFrame(rows)


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


def check_windows_separate(table):
    """Stop if a test window is also in the pretraining data ('train' and 'pretrain_only' parts) (D33, D176).

    For split_windows, where windows of one capture are on both sides by design (D163). A window is
    identified by its run, capture and start sample. Without a hop, windows do not overlap (D137), so
    different start samples of one capture mean different samples; with overlapping windows (D226)
    a training and a test window can share samples, as in the protocol being followed.
    """
    key = lambda t: set(zip(t.run_id, t.capture_id, t.start))
    shared = key(table[table.split == "test"]) & key(table[table.split.isin(["train", "pretrain_only"])])
    if shared:
        raise ValueError(f"{len(shared)} windows are in both pretraining and test, e.g. {sorted(shared)[:5]}")


def random_parts(caps, test_fraction, seed):
    """A random split of the captures into 'train' and 'test' (D169): one draw with the given
    seed; round(test_fraction * n) captures go to test. No 'pretrain_only' captures."""
    order = np.random.default_rng(seed).permutation(len(caps))
    test = np.zeros(len(caps), dtype=bool)
    test[order[: int(round(test_fraction * len(caps)))]] = True
    return {"train": caps[~test].reset_index(drop=True), "test": caps[test].reset_index(drop=True),
            "pretrain_only": caps.iloc[:0]}


def split_windows(cfg, meta, transform=None):
    """Windows of the captures matching cfg['split']['captures'], split at random into 'train' and 'test'
    (D163): scikit-learn's train_test_split, stratified by class, with the config's test_fraction and
    random_state, as Zhao et al. (D167, D171, D174). Every capture is cut whole, so windows of one
    capture are on both sides. With 'per_class' in random_windows, that many windows per class are first
    drawn at random (seeded with random_state) and only those are split, as Thuan & Hong (HUST, D226). Optional 'pretrain_only' filters add unlabelled windows for pretraining
    (D126, D164). Returns {part: (X, table)} as prepare().
    """
    split, w = cfg["split"], cfg["windows"]
    if w.get("features_file"):
        raise ValueError("stored band features (D172) are for whole-capture windows, not for split_windows")

    def windows_of(filters, part):  # cut the captures, then class and group of each window from its capture
        caps = select(meta, filters)
        caps["label"], caps["group"] = LABELS[cfg["labels"]](caps)
        X, table = cut(cfg["data_root"], cfg["dataset"], caps, cfg["channels"], w["length"], w["sample_rate_hz"],
                       transform, w.get("hop"))
        key = list(zip(caps.run_id, caps.capture_id))
        label, group = dict(zip(key, caps.label)), dict(zip(key, caps.group))
        rows = list(zip(table.run_id, table.capture_id))
        table["label"] = [label[k] for k in rows]
        table["group"] = [group[k] for k in rows]
        table.insert(0, "split", part)
        table.insert(1, "row", np.arange(len(table)))
        table.insert(4, "channels", ";".join(cfg["channels"]))
        return X, table

    X, table = windows_of(split["captures"], "all")
    if table.label.isna().any():
        raise ValueError("split['captures'] contains captures without a class; every window must have one")
    r = split["random_windows"]
    if "per_class" in r:  # D226: a fixed number of random windows per class, drawn before the split
        rng = np.random.default_rng(r["random_state"])
        keep = np.sort(np.concatenate([rng.choice(np.flatnonzero(table.label == c), r["per_class"], replace=False)
                                       for c in sorted(table.label.unique())]))
        X, table = X[keep], table.iloc[keep].reset_index(drop=True)
    train, test = train_test_split(np.arange(len(X)), test_size=r["test_fraction"], stratify=table.label,
                                   random_state=r["random_state"])
    out = {}
    for part, idx in (("train", np.sort(train)), ("test", np.sort(test))):  # rows kept in the order they were cut
        t = table.iloc[idx].reset_index(drop=True).assign(split=part)
        t["row"] = np.arange(len(t))
        out[part] = (X[idx], t)
    if "pretrain_only" in split:
        out["pretrain_only"] = windows_of(split["pretrain_only"], "pretrain_only")
    else:
        out["pretrain_only"] = (np.empty((0,) + X.shape[1:]), table.iloc[:0].assign(split="pretrain_only"))
    return out


def part_captures(cfg, meta):
    """Captures of each part, with label and group. cfg['split'] holds either filters per part
    (D147), or 'random' (D169): the labelled captures matching the filters in 'captures', split at random."""
    split = cfg["split"]
    if "random" in split:
        caps = select(meta, split.get("captures", {}))
        caps["label"], caps["group"] = LABELS[cfg["labels"]](caps)
        caps = caps[caps.label.notna()].reset_index(drop=True)  # captures without a label are left out (Herning: D55)
        return random_parts(caps, split["random"]["test_fraction"], split["random"]["seed"])
    parts = {}
    for part in PARTS:
        parts[part] = select(meta, split[part])
        parts[part]["label"], parts[part]["group"] = LABELS[cfg["labels"]](parts[part])
    return parts


def prepare(cfg, meta):
    """Windows of the 'train', 'test' and 'pretrain_only' parts of cfg['split'].

    Returns {part: (X, table)}; each table has split, row (index in X), run_id, capture_id,
    channels, start, stop, resampled_from_hz, label and group. 'train' and 'test' must have a
    label for every window; 'pretrain_only' is used without labels and may be empty.
    Parts may not share captures, except with 'random_windows' in the split (split_windows, D163),
    where they may not share windows. With windows: features (band or mssp), X holds the features of
    each window instead of its samples (D165, D181); with features_file they are read from the stored file.
    """
    w = cfg["windows"]
    transform = None
    if w.get("features"):  # D165, D181: (n, length, channels) -> features, e.g. (n, 200 windows of 1 ms, 48)
        transform = lambda X: FEATURE_SETS[w["features"]].compute(X, cfg["channels"], w["sample_rate_hz"])
    if "random_windows" in cfg["split"]:  # windows split at random, not captures (D163)
        return split_windows(cfg, meta, transform)
    out = {}
    for part, caps in part_captures(cfg, meta).items():
        if part != "pretrain_only" and caps.label.isna().any():
            raise ValueError(f"'{part}' contains captures without a class: {caps[caps.label.isna()].run_id.tolist()[:5]}")
        if len(caps) == 0:  # e.g. no pretraining-only captures in a random split; filled in below
            out[part] = None
            continue
        if w.get("features_file"):  # stored features (D172, D181) instead of computing them here
            X, table = cut_stored(w["features_file"], w["features"], caps, cfg["channels"], w["length"], w["sample_rate_hz"])
        else:
            X, table = cut(cfg["data_root"], cfg["dataset"], caps, cfg["channels"], w["length"], w["sample_rate_hz"],
                           transform, w.get("hop"))
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
    X0, t0 = next(v for v in out.values() if v is not None)
    for part in out:  # an empty part: no windows, same shape and columns as the others
        if out[part] is None:
            out[part] = (np.empty((0,) + X0.shape[1:]), t0.iloc[:0].assign(split=part))
    used = [set(zip(t.run_id, t.capture_id)) for _, t in out.values()]
    if used[0] & used[1] or used[0] & used[2] or used[1] & used[2]:
        raise ValueError("the parts of the split share captures")
    return out


def condition_codes(table, meta, columns):
    """Operating-condition classes of every pretraining window, for methods that use them (D198).

    table: the window table of a run (train, test and pretrain_only rows); the pretraining
    windows are the 'train' rows followed by the 'pretrain_only' rows, the order of X_pre in
    scripts/pretrain.py. columns: capture columns (captures or telemetry table), one per factor.
    Each factor's values become classes 0..K-1 in sorted order (discrete levels, D197).
    Returns codes (n_pretraining_windows, n_factors) and {column: [value of class 0, 1, ...]}.
    """
    pre = pd.concat([table[table.split == "train"], table[table.split == "pretrain_only"]], ignore_index=True)
    caps = select(meta, {})[["run_id", "capture_id"] + list(columns)]
    values = pre[["run_id", "capture_id"]].merge(caps, on=["run_id", "capture_id"], how="left", validate="many_to_one")
    if values[list(columns)].isna().any().any():
        raise ValueError(f"pretraining windows without a value for {list(columns)}")
    codes, classes = [], {}
    for c in columns:
        levels = np.unique(values[c].to_numpy())
        codes.append(np.searchsorted(levels, values[c].to_numpy()))
        classes[c] = [v.item() if hasattr(v, "item") else v for v in levels]
    return np.stack(codes, axis=1), classes
