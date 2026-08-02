# ADR-002: Remove the unused exposure extension

## Status

Accepted; supersedes the exposure portion of ADR-001.

## Context

The native LIC2 `renderer.cpp` accepts a trained-exposure argument but does not
apply it to rendered RGB. The Python adapter had added a trainable 3x4 affine
parameter as an experiment-specific extension. That parameter changes the
optimization problem and is not part of the effective LIC2 mapping path.

## Decision

Remove exposure from `GaussianMap`, `TrainingConfig`, the optimizer, CLI, loss
path, checkpoints, reports, and final artifacts. Existing exposure checkpoints
do not need to remain loadable during this repository's experimental phase.

## Consequences

RGB supervision now consumes the CUDA rasterizer output directly, matching the
effective native LIC2 behavior. Evaluation and checkpoint schemas no longer
carry an exposure field; old experimental checkpoints must be regenerated.
