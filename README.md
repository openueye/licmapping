licmap 是这个代码库的运行环境
3dgs_train 只是项目约定的 pytest 环境

# LIC mapping CUDA adapter

This repository is an experimental Python/PyTorch LIC2 mapping pipeline. Its
default renderer is SAGE's vendored depth-capable CUDA rasterizer as an
explicit approximate substitution because the original LIC binding currently
produces invalid RGB/alpha output in the working PyTorch environment. This
does not claim numerical or CUDA-kernel equivalence with Gaussian-LIC.

The adapter intentionally does not copy or modify the reference kernels. The
build reads the reference source tree through `LIC_REFERENCE_SRC` (default:
`../../06_CodeRefference/Gaussian-LIC/src`) and compiles a small PyTorch
extension containing the reference C++ autograd wrapper plus its CUDA sources.

## Build

Build the SAGE rasterizer into the same Conda environment used to run this
repository:

```bash
conda activate licmap
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  python -m pip install --no-build-isolation --no-deps \
  /home/DL/Projects/02_Thesis/00_Baselines/SAGE/third_party/diff-gaussian-rasterization-w-depth
```

The original LIC extension remains available for isolated diagnostics, but it
is not used by the default `lic_mapping.render` path.

The legacy LIC extension can still be built for comparison with the project
CUDA environment and local GPU architecture:

```bash
conda activate 3dgs_train
export TORCH_CUDA_ARCH_LIST=8.9
python setup.py build_ext --inplace
python -m pytest -q
```

An editable install is also supported with
`python -m pip install -e . --no-build-isolation`.

## YAML experiment configurations

The visible input, output, training, SPNet, and evaluation parameters are
stored under `config/`. Four ready-to-run Downtown1 configurations select SH
degree 0, 1, 2, or 3:

```bash
python -m lic_mapping.trainer --config config/downtown1_sh0.yaml
```

Select another degree by changing the YAML filename. The explicit CLI options
remain available and override the corresponding YAML values, so a quick test
can be limited without editing the experiment file:

```bash
python -m lic_mapping.trainer \
  --config config/downtown1_sh0.yaml \
  --frame-limit 3 \
  --iterations 1 \
  --keyframe-every 1 \
  --no-artifacts
```

Relative paths in a YAML file are resolved relative to that file's directory.
The generated report records `config_file` for experiment provenance.

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
frames accumulate source points, only every fifth frame is a keyframe, the
first keyframe initializes the map, later keyframes extend it after current-view
pixel/depth deduplication and rendered-alpha `< 0.99` gating, and opacity
pruning is disabled by default. An explicit per-keyframe new-point cap or
pruning interval can still be supplied for controlled ablations.

Gaussians use configurable degree-0/1/2/3 SH (degree 3 by default), isotropic
`z / focal` scale, opacity `0.1`, and identity rotation. Incremental extension uses LIC2's current-window
pixel/depth winner selection only; no global voxel deduplication is applied.
The renderer substitution is isolated to the backend and is recorded in each
training report as `renderer_alignment: approximate_substitution`: SAGE
receives the unchanged LIC `dc + sh_rest` tensors through its SH path, and its
silhouette pass supplies the depth/alpha contract.

```bash
python -m lic_mapping.trainer \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/lic_mapping/checkpoint.pt \
  --resize-width 800 --resize-height 648 \
  --iterations 100 \
  --sh-degree 0
```

Run the same command with `--sh-degree 1`, `--sh-degree 2`, and
`--sh-degree 3` to compare the four color models. Use separate checkpoint and
artifact paths for each run.

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
and 20 m depth limit. The SAGE Large-CNX checkpoint can be loaded directly:

```bash
python -m lic_mapping.trainer \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/lic_mapping/checkpoint.pt \
  --spnet-weights 00_Baselines/SAGE-models/Large_300.pth
```

The loader verifies the locked SPNet source tree and checkpoint SHA-256 before
loading the `{"network": state_dict}` payload. A TensorRT engine or TorchScript
export remains available through `--spnet-engine` or `--spnet-torchscript`; the
three backends are mutually exclusive. Supplying no model leaves completion
disabled and is recorded as such in the report.

Images without an interpolable odometry pose, such as the startup gap in
some bags, remain rejected slots rather than being compressed out of the
five-slot window. `RosbagReader.frames()` is a generator: image/cloud payloads
are loaded from SQLite rows on demand, and the trainer retains only the current
accumulation plus keyframe cameras. The CLI reports pose and source-slot
rejection counts in its JSON output and records them in the checkpoint report;
image, point-cloud, and calibration decode errors remain hard failures.

For mapping-core comparisons, `BagFrame` is the canonical frame-level input
contract: the same `rgb`, metric `depth_m`, world-frame `points_world` and
`point_colors`, `world_from_camera` pose, and camera intrinsics must be replayed
to each core. The deterministic replay in
`tests/fixtures/fixed_mapping_frames.json` exercises this contract without
re-reading or re-fusing the ROSBAG.

At the end of a CLI run, `<output-stem>_artifacts/` contains `metrics.json`
(per-keyframe and aggregate PSNR/SSIM/depth/alpha metrics), rendered and target
RGB PNGs, depth arrays/colour maps, alpha and absolute-error maps, and
`map/gaussians.npz` plus a standard 3DGS `point_cloud.ply`. The PLY contains
the raw DC/SH coefficients, opacity logits, log-scales and rotations, with
`f_rest_*` generated according to the selected `sh_degree`. The retained-keyframe
metrics use SAGE's offline AlexNet LPIPS, PSNR, and Gaussian-window SSIM
protocol. The AlexNet checkpoint is loaded from
`00_Baselines/SAGE-models/alexnet-owt-7be5be79.pth` by default and can be
overridden with `--lpips-backbone`.

The centered-five contract is taken from the workspace SAGE data path, while
the map lifecycle, SPNet selection, and final evaluation layout follow the
LIC2 reference. Exposure optimization is intentionally absent: the native
LIC2 renderer does not apply its unused exposure argument, so the Python
adapter keeps RGB loss and checkpoints free of an exposure parameter.

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
connected to PyTorch autograd through SAGE's CUDA backward path. The training
report records the active backend as `sage.diff_gaussian_rasterization`, with
`renderer_alignment: approximate_substitution` and
`renderer_reference: Gaussian-LIC/src/rasterizer`.

The upstream `duplicateWithKeys` kernel has a one-row edge case: do not call
this adapter with fewer than two Gaussian rows. A normal scene is far above
that limit; a pruning loop should stop before reaching one row or handle that
case outside the reference rasterizer.

The SAGE adapter parity test can be run with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n licmap python -m pytest -q \
  tests/test_mapping_contract.py tests/test_sage_parity.py
```

It compares RGB, depth, alpha, radii, and gradients against direct calls to
the rebuilt SAGE extension. This is adapter-to-SAGE parity only; it is not a
claim of numerical parity with the Gaussian-LIC rasterizer.

## Source and license boundary

The referenced Gaussian-LIC files carry their upstream GPLv3 notices. This
repository contains the binding and adapter only; it does not relicense the
referenced implementation. Keep the reference source and its license notices
available when building or distributing this experiment.
