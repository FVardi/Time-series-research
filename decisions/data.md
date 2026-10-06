# Decisions — data layer

Ingestion, canonical storage and metadata. Index: `../DECISIONS.md`.

## 2026-09-30 — D1: Data hierarchy run → capture → stream → channel; windowing downstream
**Decision:** Adapters expose whole captures, never pre-cut windows. Data is organised as run (one test, e.g. one Herning file) → capture (one uninterrupted acquisition, e.g. a 0.2 s scope sweep or a 0.93 s microphone sample; a continuously recorded dataset has one capture per recording) → stream (channels sharing a clock and sample rate, e.g. AE+UL+SP at 2.5 MHz; the two microphones at 80 kHz) → channel. Windowing is a separate, configurable stage (window length is an experimental variable); windows never span capture or stream boundaries. Combining streams with different clocks is an explicit pipeline step with a stated time tolerance, never done at ingestion.
**Alternatives considered:** Adapters returning pre-cut windows (fixes window length at ingestion). Treating every channel as an independent recording (simpler, but loses which sensors were co-recorded, which multimodal experiments must then reconstruct).
**Rationale:** Herning data is burst-sampled (0.2 s every 12 s), so the capture is the natural contiguous unit. The harness must handle multichannel and multimodal series, with streams at different rates and timing precision.

## 2026-09-30 — D2: Metadata as four tables
**Decision:** Metadata is stored in four tables. *runs*: dataset, run ID, bearing, lubricant, rig configuration, status (D7), source file and checksum, free-form `extras`. *captures*: capture ID, run, stream, start time, timing accuracy, sample count. *channels*: name, stream, sample rate, units, scale, offset, sensor type (may be `unknown`). *telemetry*: operating quantities (e.g. speed, temperature, controller state), either one row per capture or as timestamped series at native rate. Aggregation of telemetry to window level happens in label derivation, not at ingestion.
**Alternatives considered:** A single flat per-recording table with fixed fields plus `extras` (the earlier D2 draft); does not represent multiple streams or per-capture telemetry cleanly.
**Rationale:** Information naturally lives at these four levels. New sensors or rig configurations add rows, not schema changes. The timing-accuracy field preserves caveats such as the microphones' sub-second clock alignment.

## 2026-09-30 — D3: Canonical format — Parquet metadata, HDF5 signals, one file per run
**Decision:** Each source dataset is converted once to a canonical, read-only format: Parquet for the metadata tables, HDF5 for signals, one HDF5 file per run.
**Alternatives considered:** Zarr (better for parallel and cloud/object-storage access, but many small files, mitigated by v3 sharding; worth it mainly for cluster or cloud training).
**Rationale:** Simple to manage on Windows, same format as the Herning source, and HDF5's weakness with concurrent writes is irrelevant for write-once data; parallel reads work with one handle per worker. The reader interface hides the format, so switching later means re-running the conversion.

## 2026-09-30 — D4: Minimal adapter interface
**Decision:** An adapter provides exactly three things: (a) the metadata tables, (b) a signal slice (capture, channel(s), start, stop), (c) telemetry. Everything else happens downstream.
**Alternatives considered:** Adapters that also window, normalise or derive labels.
**Rationale:** Keeps dataset-specific code small; every transformation lives in shared, configurable pipeline stages.

## 2026-09-30 — D5: No transformation at ingestion; immutable canonical data
**Decision:** Values are stored as recorded, with units and calibration in metadata. No scaling, filtering or resampling at ingestion; all transformations are transparent, configurable pipeline steps. Canonical data is immutable and tagged with a checksum of its source files.
**Alternatives considered:** Normalising or resampling during conversion.
**Rationale:** Preprocessing choices are experimental variables and must be visible and reversible; checksums tie every result to its exact source data.

## 2026-09-30 — D6: Store raw integer codes for quantized sources
**Decision:** For quantized sources (e.g. the 8-bit oscilloscope in Herning), store the integer ADC codes plus per-channel scale and offset, and drop explicit time arrays (reconstructed from start time and sample rate). Sources that are natively floating point (e.g. the microphones) stay float.
**Alternatives considered:** Keep float64 voltages and time arrays as in the source (about 8× larger, no additional information).
**Rationale:** Lossless: Herning voltages are exact multiples of the scope's step size and the time arrays equal origin + i·increment. Closer to "as recorded" (D5) and reduces one Herning run from about 95 GB uncompressed to about 12 GB (int16) or 6 GB (int8).

## 2026-09-30 — D7: Dataset status field
**Decision:** Every run carries a status: `validation`, `provisional` or `final`. Reporting flags or refuses results computed on non-`final` runs.
**Alternatives considered:** Tracking data validity outside the harness (notes, file names).
**Rationale:** The current Herning run is a validation run from an unfinished rig; the status prevents such data from entering reported results by mistake.

## 2026-10-06 — D20: Herning scope data stored as 16-bit integer codes
**Decision:** The oscilloscope channels are stored as int16 codes exactly as the scope delivers them (8-bit ADC values in a 16-bit word), with per-channel scale and offset; voltage = code × scale + offset. Refines D6.
**Alternatives considered:** int8 (store only the 256 actual levels; half the uncompressed size, but only valid while the scope delivers 8-bit data, e.g. not in high-resolution acquisition mode).
**Rationale:** Closest to "as recorded" (D5) and covers future higher-resolution acquisitions. Storage is not a constraint, and compression removes most of the size difference.

## 2026-10-06 — D21: HDF5 chunk size of 16,384 samples
**Decision:** Signal datasets in canonical HDF5 files are chunked in blocks of 16,384 samples.
**Alternatives considered:** 4,096; 65,536; 250,000; 500,000 (whole sweep). Tested on 40 Herning sweeps (int16, gzip level 4): size 0.51–0.55 MB per sweep for all; random 4,096-sample window reads 0.14–2.2 ms; whole-sweep reads 2.3–3.1 ms.
**Rationale:** Best balance in the test: fast reads of short windows, near-minimal file size, whole-sweep reads only marginally slower. Chunk size does not affect stored values, only size and speed.
