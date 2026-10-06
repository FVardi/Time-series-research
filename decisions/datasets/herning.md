# Decisions — Herning test rig

Dataset-specific decisions for the AU Herning test rig (the MSSP rig data). Index: `../../DECISIONS.md`.

## 2026-10-03 — D12: κ computed as in the MSSP paper
**Decision:** The regression target κ is computed from measured speed and temperature using exactly the computation from the MSSP paper (viscosity–temperature relation, required-viscosity formula and constants). Its inputs and constants (lubricant reference viscosities, bearing mean diameter) are stored in run metadata, not hard-coded. Anything the MSSP computation does not settle is decided separately: which speed and temperature signals are used, how per-sweep telemetry is assigned to windows, which sweeps are excluded (e.g. 0 rpm, unsettled temperature after set-point changes), and the target form (κ, log κ or lubrication regimes).
**Alternatives considered:** A new κ formulation for the harness (free to improve, but results would no longer be comparable with the MSSP paper).
**Rationale:** Keeps harness results directly comparable with published work and avoids silently changing the target definition.
