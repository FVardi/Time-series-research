# Synthetic test bed (METHOD_IDEAS M5)

Toy generator with known factors, for testing the split-representation idea (D196): speed →
shaft harmonics; lubrication state → bursts; temperature → noise level; bearing → gain and
resonance. Three designs (`synthetic_crossed`, `synthetic_confounded`, `synthetic_departures`),
144 runs each, two sensors (AE-like, US-like), written by `scripts/make_synthetic.py` from
`configs/synthetic/*.yaml`; labels `synthetic`, `synthetic_shared`, `synthetic_kappa`.

Used by the parked pilot (`studies/split_pilot/`, D198, D199). Not a benchmark dataset.

History: D196, D198, D199.
