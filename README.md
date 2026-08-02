# LIC mapping CUDA adapter

This repository is an experimental Python/PyTorch seam around the CUDA
rasterizer used by `06_CodeRefference/odin_gaussian_lic`.

The adapter intentionally does not copy or modify the reference kernels. The
build reads the reference source tree through `LIC_REFERENCE_SRC` (default:
`../../06_CodeRefference/odin_gaussian_lic/src`) and compiles a small PyTorch
extension containing the reference C++ autograd wrapper plus its CUDA sources.

## Build

Use the project CUDA environment and target the local GPU architecture:

```bash
conda activate 3dgs_train
export TORCH_CUDA_ARCH_LIST=8.9
python setup.py build_ext --inplace
python -m pytest -q
```

An editable install is also supported with
`python -m pip install -e . --no-build-isolation`.

## Fixed-pose ROSBAG training

The mapping input reads the Odin ROS2 bag topics
`/odin1/image/compressed`, `/odin1/odometry`, and `/odin1/cloud_slam`, performs
FishPoly rectification and strict pose interpolation, and keeps the original
image-slot sequence fixed. Five consecutive finalized source slots (`t-2` to
`t+2`) are projected into the target camera with a nearest-depth z-buffer over
`[0.1, 200]` m. The minimum valid depth is fused, conflicts larger than 1 m
are rejected, and valid center depth is restored with center priority.

The resulting mapping depth uses the center source wherever available and the
centered-five source only to fill center holes. Its source type and confidence
(`1.0` for center, `0.7` for fused-five) are retained through Gaussian
initialization and append. The LIC2 lifecycle is reproduced: all accepted
frames accumulate source points, only every eighth frame is a keyframe, the
first keyframe initializes the map, later keyframes extend it after current-view
pixel/depth deduplication and rendered-alpha `< 0.99` gating, and opacity
pruning runs every five keyframes at threshold `0.01`.

Gaussians use degree-3 SH, isotropic `2 * z / focal` scale, opacity `0.1`,
identity rotation, and the requested outer first-wins 5 cm `floor` voxel
deduplication:

```bash
python -m lic_mapping.trainer \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/lic_mapping/checkpoint.pt \
  --resize-width 800 --resize-height 648 \
  --iterations 30
```

Enable the native LIC2 SPNet completion path by supplying the matching
TensorRT engine (the engine resolution must equal the resized image):

```bash
python -m lic_mapping.trainer \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/lic_mapping/checkpoint.pt \
  --resize-width 800 --resize-height 648 \
  --spnet-engine /path/to/spnet_800_648.engine
```

SPNet is invoked only on keyframes. Its metric output follows LIC2's
`depth / 200` input scale and is converted into sparse blind-region points by
the native 10-pixel patch selection, known-point bias check, Sobel edge gate,
and 20 m depth limit. A TorchScript export can be used with
`--spnet-torchscript`; the two backends are mutually exclusive. Supplying no
model leaves completion disabled and is recorded as such in the report.

Exposure optimization is enabled by default. It trains LIC2's identity-
initialized 3x4 affine exposure matrix with learning rate `0.001`, stores the
matrix in the checkpoint, and applies it before the photometric loss. Use
`--no-exposure` for the uncorrected baseline.

Images without an interpolable odometry pose, such as the startup gap in
some bags, remain rejected slots rather than being compressed out of the
five-slot window. `RosbagReader.frames()` is a generator: image/cloud payloads
are loaded from SQLite rows on demand, and the trainer retains only the current
accumulation plus keyframe cameras. The CLI reports pose and source-slot
rejection counts in its JSON output and records them in the checkpoint report;
image, point-cloud, and calibration decode errors remain hard failures.

At the end of a CLI run, `<output-stem>_artifacts/` contains `metrics.json`
(per-keyframe and aggregate PSNR/SSIM/depth/alpha metrics), rendered and target
RGB PNGs, depth arrays/colour maps, alpha and absolute-error maps, and
`map/gaussians.npz` plus a binary `point_cloud.ply`. LPIPS is computed when a
compatible TorchScript model is supplied with `--lpips-model`; otherwise the
artifact explicitly records that LPIPS was not requested.

The centered-five contract is taken from the workspace SAGE data path, while
the map lifecycle, SPNet selection, exposure parameter, and final evaluation
layout follow the LIC2 reference.

The public Python interface is deliberately small:

```python
from lic_mapping import LicCamera, render

output = render(
    means3d, dc, sh, opacities, scales, rotations,
    LicCamera(width, height, fx, fy, cx, cy, world_from_camera),
    sh_degree=3,
)
```

`output.rgb` is `[3, H, W]`, `output.depth` is `[H, W]`, `output.radii` is
`[N]`, and `output.visible` is `output.radii > 0`. The returned tensors remain
connected to PyTorch autograd through the reference CUDA backward path.

The upstream `duplicateWithKeys` kernel has a one-row edge case: do not call
this adapter with fewer than two Gaussian rows. A normal scene is far above
that limit; a pruning loop should stop before reaching one row or handle that
case outside the reference rasterizer.

## Source and license boundary

The referenced Gaussian-LIC files carry their upstream GPLv3 notices. This
repository contains the binding and adapter only; it does not relicense the
referenced implementation. Keep the reference source and its license notices
available when building or distributing this experiment.
