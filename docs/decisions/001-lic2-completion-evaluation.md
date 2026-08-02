# ADR-001: Keep LIC2 completion and evaluation behind explicit adapters

## Status

Accepted

## Context

The Python mapping baseline already owns the LIC CUDA rasterizer and the fixed-
pose streaming ROSBAG lifecycle. LIC2 additionally performs keyframe-only
SPNet depth completion, optimizes a global 3x4 exposure matrix, and writes
training-view quality/visualization artifacts. The workspace does not contain
a distributable SPNet engine or weights, so silently using an approximation
would make an end-to-end result non-comparable.

## Decision

- Use a `DepthCompleter` protocol with native TensorRT and TorchScript adapters.
- Require an explicit engine/model path to enable SPNet; no implicit fallback is
  used in production runs.
- Apply LIC2's keyframe completion selection in Python: known-depth bias check,
  Sobel edge gate, 10-pixel patches, nearest completed depth per patch, and a
  20 m limit. Completed points carry the SAGE `SPNET_BLIND` source identity.
- Store and optimize LIC2's identity-initialized 3x4 exposure matrix at
  learning rate `0.001`.
- Apply the matrix as an identity-plus-residual transform. This preserves the
  exact initial RGB path and avoids a LIC rasterizer NaN observed on the first
  multi-keyframe Downtown1 smoke when an identity matrix was evaluated through
  a standalone einsum.
- Make final artifacts deterministic and self-describing: raw arrays and a
  binary PLY accompany PNG visualizations, while optional LPIPS is explicitly
  marked unavailable when no compatible model is provided.

## Consequences

The baseline can run and test without proprietary/large SPNet assets, but a
real SPNet experiment must provide a matching model and its provenance. The
retained keyframe views are enough for LIC2-style final evaluation without
materializing all ROSBAG frames. Checkpoints now include exposure state and
reports include completion/evaluation configuration.
