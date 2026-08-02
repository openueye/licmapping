licmap 是运行环境；`3dgs_train` 仅用于 pytest。

# LIC mapping CUDA adapter

Python/PyTorch fixed-pose LIC2 mapping baseline。默认使用 SAGE CUDA
rasterizer，作为 Gaussian-LIC 原生 rasterizer 的近似替代，不宣称数值等价。

## Run

```bash
conda activate licmap
CUDA_HOME=/usr/local/cuda-12.9 \
PATH="/usr/local/cuda-12.9/bin:$PATH" \
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=1 \
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  /home/DL/Projects/02_Thesis/00_Baselines/SAGE/third_party/diff-gaussian-rasterization-w-depth

python -m lic_mapping.trainer --config config/downtown1_sh0.yaml
```

`downtown1_sh0.yaml`–`downtown1_sh3.yaml` 对应 SH degree 0–3。

默认 mapping 参数：`keyframe_every=5`、`iterations_per_frame=100`、关闭
pruning、无 Gaussian 数量上限、`scale_multiplier=1.0`。单次新增上限使用
`max_new_points_per_frame` / `--max-new-points`。

## Input contract

`RosbagReader` 将 Odin ROS2 的 image、odometry、`cloud_slam` 转为 `BagFrame`。
mapping core 比较必须重放相同的 `rgb`、metric `depth_m`、world-frame
`points_world`/`point_colors`、pose 和 intrinsics。固定 fixture：
`tests/fixtures/fixed_mapping_frames.json`。

前端采用 centered-five (`t-2`…`t+2`) depth z-buffer，中心深度优先；无效
pose/source 的 slot 保留为 rejected。

## Optional SPNet

默认关闭 depth completion。可通过 `--spnet-engine`、`--spnet-torchscript` 或
`--spnet-weights` 启用，三者互斥；SPNet 只在 keyframe 上运行。

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n licmap python -m pytest -q
```

Parity 测试比较 adapter 与 direct SAGE extension 的 RGB、depth、alpha、radii
和 gradient：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n licmap python -m pytest -q \
  tests/test_mapping_contract.py tests/test_sage_parity.py
```

## Outputs and limits

运行结果写入 `<output-stem>_artifacts/`，包括 metrics、RGB/depth/alpha 可视化、
Gaussian arrays 和 3DGS PLY；报告记录 renderer alignment 和输入拒绝数。

当前 parity 只覆盖 SAGE adapter；Gaussian-LIC 原生 mapping executable 尚未接入
`BagFrame` replay。rasterizer 至少需要两个 Gaussian row。参考源码位于
`../../06_CodeRefference/Gaussian-LIC/src`，其 GPLv3 notices 必须保留。
