# Decisions — training

Method integration, pretraining and the training loop. Index: `../DECISIONS.md`.

## 2026-10-03 — D10: Two ways to integrate a method, one shared encoder boundary
**Decision:** The harness supports two kinds of method. (A) *Wrapped methods*: official code used as a black box behind an adapter that trains on unlabelled training data and exposes `encode(X)`; architecture, augmentations and loss are left untouched. (B) *Composed methods*: built from harness components — tokenizer, encoder, view generator (augmentations or masking), objective and a shared training loop — combined through configuration. Both produce the same artifact (an encoder checkpoint or stored embeddings), so evaluation is independent of how the encoder was trained. Safeguards for (B): (1) every composed counterpart of a published method must reproduce the official (A) result within the seed spread on one benchmark before it is used; (2) reimplemented components are tested against the official functions (same input and seed, same output); (3) components have no internal defaults — every parameter is set in the configuration; (4) the comparisons a study will make are written down before it is run. Which methods to implement first, and in which mode, is decided separately.
**Alternatives considered:** Wrapped methods only (faithful to published results, but no controlled swapping of objectives or views; Steps 2.4 and 3 of the research plan would require restructuring). Composed methods only (flexible, but no reference implementation to validate reimplementations against, and weaker comparability with published results).
**Rationale:** Step 1 of the research plan needs faithful baselines; Steps 2–3 need controlled comparisons in which only one component differs. A shared boundary lets both coexist. The added flexibility of (B) creates room for silent reimplementation errors, hidden defaults, invalid combinations and selective reporting; the safeguards address these.

## 2026-10-06 — D15: Plain PyTorch; scikit-learn for probes and simple baselines
**Decision:** Models and training loops are written in plain PyTorch, with the training loop written out explicitly. scikit-learn is used for probes and simple baselines.
**Alternatives considered:** PyTorch Lightning (less boilerplate, but hides the training loop).
**Rationale:** Readable, checkable code (code-style rule in CLAUDE.md).

## 2026-10-06 — D19: Official method code as pinned git submodules; modified versions as separate, named methods
**Decision:** Official method repositories (e.g. T-Loss, TS2Vec) are added under `external/` as git submodules pinned to an exact commit and are never edited. Our adapter in `harness/methods/` calls them. A fix needed only to make the code run (e.g. a changed library API) goes in the adapter, documented in a comment. A fix that must change the method's own code, or an improvement, is made in a copy under `harness/methods/`, under its own method name, with a header stating the original repository and commit and what was changed. The unmodified original stays available as the reference, and the modified version is validated against it (D10). Results from a modified version are always reported under its own name, never as the original method.
**Alternatives considered:** Copying official code into the repository (loses provenance; invites quiet edits). Installing from git at a pinned commit (clean, but most research repositories are not installable packages). Editing the submodule directly (mixes our changes with theirs and hides them).
**Rationale:** Records exactly which version was used, keeps the original as a reference for validation, and makes every modification visible and attributable.

## 2026-10-07 — D28: TS2Vec training settings for the first slice
**Decision:** TS2Vec is trained with the official defaults, all written explicitly in the config: learning rate 0.001, batch size 8, representation dimension 320, hidden dimension 64, depth 10, maximum training length 3,000, and the number of iterations given by TS2Vec's own rule (200 if the training data has at most 100,000 values, otherwise 600; 600 for FordA).
**Alternatives considered:** Tuning TS2Vec's settings (would no longer reproduce the published setting).
**Rationale:** The first slice must reproduce the published result before anything else.

## 2026-10-08 — D32: T-Loss on FordA
**Decision:** T-Loss is wrapped from its official repository (submodule `external/tloss`, commit 4aff592), using only its encoder training and encoding; evaluation runs through the harness probes (D26), whose SVM probe is the protocol behind T-Loss's published results. Settings (official, all in the config): batch size 10, learning rate 0.001, depth 10, 40 channels, kernel size 3, K = 10 negative samples, full-length comparisons, no early stopping, representation size 320 (internal size 160, as in the code and README; the paper swaps them), 2,000 training steps (paper, for K ≥ 10; the shipped default file says 1,500). T-Loss trains in 64-bit floats, as its code requires. Reproduction target: 0.928 accuracy on FordA (K = 10), with the D26 criterion; same probes, label budgets and seeds as D26. FordA as converted (D27).
**Alternatives considered:** 1,500 steps as in the default file (not the setting of the published number). Using T-Loss's own classifier code (relies on a scikit-learn option that has since been removed).
**Rationale:** A second reproduction check of a wrapped method on a dataset we already have, with a published number to compare against.
