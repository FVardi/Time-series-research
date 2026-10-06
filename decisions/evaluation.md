# Decisions — evaluation

Evaluation tasks, protocols, probes and metrics. Index: `../DECISIONS.md`.

## 2026-10-06 — D13: Tests are chosen per question and per dataset; full grid within a study; built incrementally
**Decision:** Not every test is run for every method. Each test must answer a stated research question. Tasks (regression, classification, anomaly detection) follow from each dataset's labels; protocols (frozen probe, fine-tuning, transfer to an unseen domain) are used only where they answer a question (e.g. fine-tuning where comparison with published results requires it; transfer only on datasets built for it). Within a study, the test grid is fixed in the study plan before running, and every method fills every cell; a cell a method cannot run is reported as missing with the reason, never dropped. The grid is built incrementally: comparisons are added only when shown to be necessary. The Step 1 grid discussed on 2026-10-06 is preliminary and not decided.
**Alternatives considered:** Running every task and protocol for every method and dataset (full factorial; hundreds of runs, most answering no specific question). Choosing tests per method (invites cherry-picking).
**Rationale:** Keeps comparisons fair and the number of runs proportionate to the questions asked.
