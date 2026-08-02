`licmap` 是唯一运行和测试环境；不使用 `3dgs_train`。

# Gaussian-LIC native mapping adapter

Python/PyTorch fixed-pose Gaussian-LIC mapping baseline。训练、扩图和评估
统一调用 Gaussian-LIC 原生 CUDA rasterizer；SAGE rasterizer 不属于正式路径。

## Install

```bash
conda env create -f environment.yml
conda activate licmap
CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=1 \
python -m pip install --no-build-isolation --no-deps --force-reinstall .
```

## Run

```bash
cp config/downtown1.local.example.yaml config/downtown1.local.yaml
python -m lic_mapping.trainer --config config/downtown1.local.yaml
```

`downtown1_sh0.yaml`–`downtown1_sh3.yaml` 是冻结的正式复现配置（SH degree
0–3），不会被本机路径覆盖。`downtown1.local.yaml` 已为当前工作区生成并被
Git 忽略；修改其 `training.sh_degree` 或使用 `--sh-degree 0..3` 运行本地实验。

默认 mapping 参数：`keyframe_every=5`、`iterations_per_frame=100`、关闭
pruning、无 Gaussian 数量上限、`scale_multiplier=1.0`。单次新增上限使用
`max_new_points_per_frame` / `--max-new-points`。

## Input contract

`RosbagReader` 将 Odin ROS2 的 image、odometry、`cloud_slam` 转为 `BagFrame`。
mapping core 比较必须重放相同的 `rgb`、metric `depth_m`、world-frame
`points_world`/`point_colors`、pose 和 intrinsics。固定 fixture：
`tests/fixtures/fixed_mapping_frames.json`。

Odin ROS2 bag 缺少参考实现的上游 depth topic。经实验授权，前端使用同一
同步 SLAM 点云的单帧 z-buffer 投影深度；不会混入邻帧。该前端在报告中标记
为 `experimental_adapter`，其余 mapping 后端为 `native_reference`。

## Optional SPNet

默认关闭 depth completion。可通过 `--spnet-engine`、`--spnet-torchscript` 或
`--spnet-weights` 启用，三者互斥；SPNet 只在 keyframe 上运行。

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n licmap python -m pytest -q
```

绑定层 parity 测试独立重建 `Camera` 的 C++ 矩阵约定，并比较 Python adapter 与
direct Gaussian-LIC binding 的 RGB、depth、transmittance、radii 和 gradient：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n licmap python -m pytest -q \
  tests/test_mapping_contract.py tests/test_native_parity.py
```

## Outputs and limits

运行结果写入 `<output-stem>_artifacts/`，包括 metrics、RGB/depth/alpha 可视化、
Gaussian arrays 和 3DGS PLY；报告记录 renderer alignment 和输入拒绝数。

SPNet 的 TensorRT engine 是 native 路径；YAML 中的已验证 `.pth` checkpoint
是显式标记的 PyTorch 实验适配。rasterizer 至少需要两个 Gaussian row。参考
源码位于 `../../06_Referrance/Gaussian-LIC/src`，其 GPLv3 notices 必须保留。
