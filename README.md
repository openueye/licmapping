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

The first end-to-end mapping loop reads the Odin ROS2 bag topics
`/odin1/image/compressed`, `/odin1/odometry`, and `/odin1/cloud_slam`, performs
FishPoly rectification and strict pose interpolation, initializes Gaussians from
the world-frame cloud, appends new voxels over time, and optimizes RGB plus
optional LiDAR depth loss while keeping all poses frozen:

```bash
python -m lic_mapping.trainer \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/lic_mapping/checkpoint.pt \
  --resize-width 800 --resize-height 648 \
  --iterations 30
```

Images without an interpolable odometry pose, such as the startup gap in
some bags, are skipped automatically. The CLI reports the skipped count in
its JSON output and records it in the checkpoint report; image, point-cloud,
and calibration errors remain hard failures.

This is intentionally a minimal backend baseline. It does not yet implement
SAGE's centered-five source fusion, SPNet prior, pruning policy, or formal
artifact receipts.

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

This first slice only validates the rasterizer seam. LIC point-cloud
accumulation, Gaussian initialization/extension, loss scheduling, sparse Adam,
and ROSBAG frame loading remain intentionally outside this module.

The upstream `duplicateWithKeys` kernel has a one-row edge case: do not call
this adapter with fewer than two Gaussian rows. A normal scene is far above
that limit; a pruning loop should stop before reaching one row or handle that
case outside the reference rasterizer.

## Source and license boundary

The referenced Gaussian-LIC files carry their upstream GPLv3 notices. This
repository contains the binding and adapter only; it does not relicense the
referenced implementation. Keep the reference source and its license notices
available when building or distributing this experiment.
