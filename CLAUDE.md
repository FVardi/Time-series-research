# CLAUDE.md

Research harness for baselining and developing self-supervised learning
methods for time series, with a focus on condition monitoring and
lubrication condition monitoring of ball bearings.

## Collaboration rules
- I make all design decisions: repo structure, architectures, model choices,
  hyperparameters and tuning procedures, training setup, preprocessing,
  label derivation, splits, configurations, evaluation methods and metrics.
- Propose options with trade-offs and a recommendation, then wait for my
  approval before implementing anything.
- Never introduce silent defaults. This includes library defaults that affect
  results (regularization strength, initialization, normalization, seeds,
  window lengths). If implementation requires an undecided choice, stop and
  ask, or leave a clearly marked placeholder and list it in your reply.
- Record every approved decision in DECISIONS.md: date, decision,
  alternatives considered, rationale.
- Purely mechanical changes (formatting, typo fixes, behaviour-preserving
  refactors) may proceed, but must be mentioned.
