# LIC Mapping 运行命令

本文档对应当前 `LIC_mapping` 的 YAML 配置入口。推荐使用独立的 `licmap` 环境运行；所有实验超参数集中在 `config/` 下的 YAML 文件中，命令行参数可以覆盖 YAML 中的值。

## 1. 激活环境

```bash
conda activate licmap
cd /home/DL/Projects/02_Thesis/00_Baselines/LIC_mapping

# 避免宿主环境中的 Python 包和 CUDA 动态库干扰
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
unset PYTHONPATH
```

## 2. SAGE rasterizer（仅首次安装）

当前默认使用 SAGE 的带 depth rasterizer。环境创建后只需安装一次：

```bash
python -m pip install --no-build-isolation --no-deps \
  /home/DL/Projects/02_Thesis/00_Baselines/SAGE/third_party/diff-gaussian-rasterization-w-depth
```

之后每次运行只需要激活 `licmap` 环境，不需要重复安装 rasterizer。

## 3. 运行测试

```bash
python -m pytest -q
```

## 4. 完整训练

配置文件已经包含 Downtown1 的 rosbag、标定文件、SAGE/LPIPS 权重、训练参数和输出路径。分别使用以下命令测试 SH degree 0、1、2、3：

```bash
CUDA_VISIBLE_DEVICES=0 python -m lic_mapping.trainer \
  --config config/downtown1_sh0.yaml

CUDA_VISIBLE_DEVICES=0 python -m lic_mapping.trainer \
  --config config/downtown1_sh1.yaml

CUDA_VISIBLE_DEVICES=0 python -m lic_mapping.trainer \
  --config config/downtown1_sh2.yaml

CUDA_VISIBLE_DEVICES=0 python -m lic_mapping.trainer \
  --config config/downtown1_sh3.yaml
```

也可以依次运行四种 SH degree：

```bash
for degree in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 \
  python -m lic_mapping.trainer \
    --config config/downtown1_sh${degree}.yaml
done
```

## 5. 快速 smoke test

下面的命令只读取 3 帧，每个关键帧优化 1 次，并关闭评估产物生成：

```bash
CUDA_VISIBLE_DEVICES=0 python -m lic_mapping.trainer \
  --config config/downtown1_sh0.yaml \
  --frame-limit 3 \
  --iterations 1 \
  --keyframe-every 1 \
  --no-artifacts
```

仅切换 SH degree 时，可以覆盖配置文件中的值：

```bash
CUDA_VISIBLE_DEVICES=0 python -m lic_mapping.trainer \
  --config config/downtown1_sh0.yaml \
  --sh-degree 3 \
  --frame-limit 3 \
  --iterations 1 \
  --keyframe-every 1 \
  --no-artifacts
```

## 6. 修改实验参数

直接编辑对应 YAML 文件，例如：

```yaml
training:
  sh_degree: 0
  iterations: 30
  keyframe_every: 8
  max_gaussians: 250000
  prune_every: 5
  prune_opacity: 0.01
```

常用配置区段如下：

- `input`：rosbag 和相机标定路径。
- `output`：checkpoint、artifact 目录和是否保存产物。
- `training`：SH degree、迭代次数、关键帧间隔、Gaussian 上限、剪枝等参数。
- `spnet`：SPNet depth completion 权重、patch size 和最大深度。
- `evaluation`：LPIPS backbone 权重与评估开关。

命令行参数优先级高于 YAML。例如：

```bash
python -m lic_mapping.trainer \
  --config config/downtown1_sh0.yaml \
  --iterations 5 \
  --frame-limit 20
```

如果不使用 SPNet depth completion，可以在 YAML 中将 `spnet` 下的权重、engine 和 TorchScript 路径设为 `null`；三种后端都为空时会自动关闭 depth completion。

## 7. 训练过程输出

终端只打印每 20 个关键帧一次，输出当前帧号、累计训练耗时和优化视图数，例如：

```text
LIC keyframe 20: frame=694, elapsed=123.45s, optimized_views=20
```

## 8. 输出文件

以 `downtown1_sh0.yaml` 为例，默认输出为：

```text
/home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_sh0.pt
/home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_sh0_artifacts/
```

标准 3DGS PLY 位于：

```text
/home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_sh0_artifacts/map/point_cloud.ply
```

该 PLY 使用标准 3DGS 字段；`f_rest_*` 的数量由 `sh_degree` 决定：degree 0、1、2、3 分别对应 0、9、24、45 个字段。
