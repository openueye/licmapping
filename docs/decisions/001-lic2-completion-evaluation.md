# ADR-001: Keep LIC2 completion and evaluation behind explicit adapters

## Status

Accepted

## Context

The Python mapping baseline already owns the LIC CUDA rasterizer and the fixed-
pose streaming ROSBAG lifecycle. LIC2 additionally performs keyframe-only
SPNet depth completion and writes training-view quality/visualization
artifacts. The workspace does not contain
a distributable SPNet engine or weights, so silently using an approximation
would make an end-to-end result non-comparable.

## Decision

- Use a `DepthCompleter` protocol with native TensorRT, TorchScript, and the
  verified SAGE Large-CNX `.pth` adapter.
- Require an explicit engine/model path to enable SPNet; no implicit fallback is
  used in production runs.
- Load `Large_300.pth` only after verifying the SAGE SPNet source-tree and
  checkpoint SHA-256 identities.
- Apply LIC2's keyframe completion selection in Python: known-depth bias check,
  Sobel edge gate, 10-pixel patches, nearest completed depth per patch, and a
  20 m limit. Completed points carry the SAGE `SPNET_BLIND` source identity.
- Make final artifacts deterministic and self-describing: raw arrays and a
  binary PLY accompany PNG visualizations, and retained-keyframe image metrics
  use SAGE's local AlexNet LPIPS protocol.

## Consequences

The baseline can run and test without large SPNet assets, but a real SPNet
experiment must provide a matching model and its provenance. The retained
keyframe views are enough for LIC2-style final evaluation without materializing
all ROSBAG frames. Exposure is deliberately absent from the experimental
checkpoint schema; see ADR-002.
