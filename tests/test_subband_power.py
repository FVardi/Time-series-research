"""Checks of the stored sub-band power spectrogram (D186)."""

import numpy as np

from harness import band_features as bf
from harness import stored_features
from harness import subband_power as sp

FS = 2.5e6


def test_subbands_tile_each_band():
    # Within each D165 band the sub-band edges run from the lower to the upper band edge, 1/12 octave
    # apart (the AE top band is cut at 560 kHz); every FFT bin of a band falls in exactly one kept
    # sub-band, so dropping the empty sub-bands loses no bin.
    f = np.fft.rfftfreq(bf.WINDOW, 1 / FS)
    for ch in ("AE", "UL"):
        sb = sp.subbands(ch, FS)
        for b, (f_lo, f_hi) in enumerate(bf.BANDS[ch]):
            e = sp.band_edges(f_lo, f_hi, sp.STEP[ch])
            assert e[0] == f_lo and e[-1] == f_hi
            assert np.allclose(np.diff(np.log(e[:-1])), np.log(sp.STEP[ch]))
            kept = [(lo, hi) for i, lo, hi in sb if i == b]
            in_band = f[(f >= f_lo) & (f < f_hi)]
            hits = [sum(lo <= x < hi for lo, hi in kept) for x in in_band]
            assert hits == [1] * len(in_band)
            assert all(np.any((f >= lo) & (f < hi)) for lo, hi in kept)
    assert len(sp.subbands("AE", FS)) == 3 * 12 + 6  # three full octaves, then 400-560 kHz in six steps
    assert sp.subbands("AE", FS)[-1][2] == 560e3


def test_a_bin_tone_lands_in_its_sub_band():
    # A sine at an FFT bin frequency (whole number of periods per window): its power A^2/2 lands in the
    # sub-band containing that frequency and nowhere else.
    t = np.arange(4 * bf.WINDOW) / FS
    frames = (3.0 * np.sin(2 * np.pi * 137e3 * t)).reshape(4, bf.WINDOW)
    edges = [(lo, hi) for _, lo, hi in sp.subbands("AE", FS)]
    P = sp.window_power(frames, FS, edges)
    k = [i for i, (lo, hi) in enumerate(edges) if lo <= 137e3 < hi][0]
    assert np.allclose(P[:, k], 4.5)
    assert np.allclose(np.delete(P, k, axis=1), 0, atol=1e-18)


def test_power_adds_up_to_the_mean_square():
    # Parseval: over all bins between DC and Nyquist, the window powers add up to the mean square of a
    # zero-mean window (up to the Nyquist bin, which no band includes).
    frames = np.random.default_rng(0).standard_normal((5, bf.WINDOW))
    frames -= frames.mean(axis=1, keepdims=True)
    P = sp.window_power(frames, FS, [(1.0, FS / 2)])
    assert np.allclose(P[:, 0], np.mean(frames ** 2, axis=1), rtol=1e-2)


def test_octave_sum_matches_band_energy():
    # The sub-bands of one octave add up to about the band signal's mean square, i.e. exp of the D165
    # log energy of the same window (white noise). A little of the band's power is spread outside the
    # band by the cut, so the sum is slightly lower. Only bands with at least 6 sub-bands are checked:
    # the lowest UL octaves hold 1-3 FFT bins at 1 ms (1 kHz spacing) and catch their band's power
    # only roughly, window by window.
    X = np.random.default_rng(1).standard_normal((1, 40 * bf.WINDOW, 2))
    P = sp.compute(X, ["AE", "UL"], FS)[0][1:-1]  # first and last window: filter transients at the sweep ends
    E = np.exp(bf.compute(X, ["AE", "UL"], FS)[0][1:-1, 0::4])  # log_energy columns, AE bands then UL bands
    col, offset = 0, 0
    for ch in ("AE", "UL"):
        sb = sp.subbands(ch, FS)
        for b in range(len(bf.BANDS[ch])):
            idx = [offset + i for i, (j, _, _) in enumerate(sb) if j == b]
            if len(idx) >= 6:
                ratio = P[:, idx].sum(axis=1) / E[:, col]
                assert 0.93 < np.median(ratio) <= 1.0 and ratio.max() <= 1.0 + 1e-9, (ch, b)
            col += 1
        offset += len(sb)


def test_strong_low_frequency_tone_does_not_leak_into_upper_bands():
    # The reason for filtering before cutting (D186 check): a strong 3.3 kHz tone, about 60 dB above the
    # noise, leaves the UL 22.4-44.7 kHz sub-bands at the noise level.
    rng = np.random.default_rng(2)
    t = np.arange(20 * bf.WINDOW) / FS
    noise = 1e-3 * rng.standard_normal(len(t))
    x_tone = np.sin(2 * np.pi * 3.3e3 * t + 0.4) + noise
    sb = sp.subbands("UL", FS)
    idx = [i for i, (j, _, _) in enumerate(sb) if j == 5]  # UL band 5: 22.4-44.7 kHz
    with_tone = sp.channel_power(x_tone, FS, "UL")[1:-1, idx].sum(axis=1)  # ends: filter transients
    without = sp.channel_power(noise, FS, "UL")[1:-1, idx].sum(axis=1)
    assert np.allclose(with_tone, without, rtol=0.05)


def test_shapes_names_and_float32_store(tmp_path):
    X = np.random.default_rng(3).standard_normal((2, 3 * bf.WINDOW + 100, 2))  # 3 full windows + a remainder
    Z = sp.compute(X, ["AE", "UL"], FS)
    n = len(sp.subbands("AE", FS)) + len(sp.subbands("UL", FS))
    assert Z.shape == (2, 3, n)
    names = sp.names(["AE", "UL"])
    assert len(names) == n and names[0] == "AE 50-52.97 kHz power" and names[-1].endswith("177.8 kHz power")
    path = tmp_path / "sub.h5"
    s = sp.settings(["AE", "UL"], FS)
    with stored_features.create(path, 2, Z.shape[1:], s, names, dtype=sp.STORE_DTYPE) as f:
        f["features"][:] = Z
        f["run_id"][:], f["capture_id"][:] = ["r", "r"], ["a", "b"]
    got = stored_features.load(path, ["r", "r"], ["b", "a"], s)
    assert np.array_equal(got, Z[::-1].astype(np.float32))
