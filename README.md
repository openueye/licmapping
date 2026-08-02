licmap 是本代码库的运行环境；`3dgs_train` 仅作为项目约定的 pytest 环境。

# LIC mapping CUDA adapter

Python/PyTorch fixed-pose LIC2 mapping baseline. 默认使用 SAGE 的 depth-capable
CUDA rasterizer，作为 Gaussian-LIC 原生 rasterizer 的近似替代；不宣称 CUDA
kernel 或数值等价。

## Quick start

```bash
conda activate licmap
CUDA_HOME=/usr/local/cuda-12.9 \
PATH="/usr/local/cuda-12.9/bin:$PATH" \
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=1 \
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  /home/DL/Projects/02_Thesis/00_Baselines/SAGE/third_party/diff-gaussian-rasterization-w-depth

python -m lic_mapping.trainer --config config/downtown1_sh0.yaml
```

四份 Downtown1 配置分别使用 SH degree 0–3：
`config/downtown1_sh0.yaml` … `config/downtown1_sh3.yaml`。

## Mapping defaults

- `keyframe_every=5`, `iterations_per_frame=100`
- 默认关闭 pruning；默认不限制初始化或单次新增 Gaussian 数量
- 可选单次新增上限：`max_new_points_per_frame` / `--max-new-points`
- 默认 `scale_multiplier=1.0`、`learning_rate_scales=0.005`、`iteration_decay=false`
- SAGE 后端在 report 中标记为 `renderer_alignment: approximate_substitution`

命令行参数会覆盖 YAML：

```bash
python -m lic_mapping.trainer \
  --config config/downtown1_sh0.yaml \
  --frame-limit 3 --iterations 1 --keyframe-every 1 --no-artifacts
```

## Input contract

`RosbagReader` 读取 Odin ROS2 的 image、odometry 和 `cloud_slam`，输出
`BagFrame`。mapping core 的比较必须重放同一组 frame-level 数据：

- `rgb`、metric `depth_m`
- world-frame `points_world` 和 `point_colors`
- `world_from_camera`、camera intrinsics

输入前端使用 centered-five (`t-2` … `t+2`) depth z-buffer，中心帧深度优先；
无有效 pose/source 的 slot 保留为 rejected，不压缩时间窗口。

固定 replay fixture：
`tests/fixtures/fixed_mapping_frames.json`。

## Optional SPNet

默认不启用 depth completion。可提供 TensorRT、TorchScript 或经过 provenance
校验的 SAGE checkpoint，三者互斥：

```bash
python -m lic_mapping.trainer \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/lic_mapping/checkpoint.pt \
  --spnet-engine /path/to/spnet_800_648.engine
```

SPNet 仅在 keyframe 上运行，并使用 LIC2 的 10-pixel patch、known-depth bias、
Sobel edge gate 和 20 m 深度限制。

## Outputs and API

运行后 `<output-stem>_artifacts/` 保存 metrics、RGB/depth/alpha 可视化、
Gaussian arrays 和 3DGS PLY。报告同时记录 renderer、alignment、输入拒绝数和
SPNet provenance。

```python
from lic_mapping import LicCamera, render

output = render(means3d, dc, sh, opacities, scales, rotations, camera,
                sh_degree=3)
```

输出为 `rgb [3,H,W]`、`depth [H,W]`、`radii [N]`；`visible` 等于
`radii > 0`，并保持 autograd 连接。

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n licmap python -m pytest -q
```

SAGE adapter parity test：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n licmap python -m pytest -q \
  tests/test_mapping_contract.py tests/test_sage_parity.py
```

测试逐项比较 adapter 与 direct SAGE extension 的 RGB、depth、alpha、radii 和
gradient；这不是 Gaussian-LIC 原生 rasterizer parity。

## Known limits

- 原生 Gaussian-LIC mapping executable 仍使用自己的 ROS topic 输入，尚未接入
  `BagFrame` replay；因此当前 parity 仅覆盖 SAGE adapter。
- rasterizer 不应接收少于两个 Gaussian row。
- 参考 Gaussian-LIC 源码位于 `../../06_CodeRefference/Gaussian-LIC/src`，其
  GPLv3 notices 必须随构建保留。
