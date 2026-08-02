# LIC mapping terminal commands

以下命令默认使用独立的 `licmap` Conda 环境，并通过 SAGE CUDA rasterizer
运行 LIC2 mapping。

## 1. 激活环境并安装 SAGE rasterizer

```bash
conda activate licmap
cd /home/DL/Projects/02_Thesis/00_Baselines/LIC_mapping

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
unset PYTHONPATH

python -m pip install --no-build-isolation --no-deps \
  /home/DL/Projects/02_Thesis/00_Baselines/SAGE/third_party/diff-gaussian-rasterization-w-depth
```

## 2. 运行测试

```bash
python -m pytest -q
```

## 3. 单个 SH degree 训练

`--sh-degree` 可设置为 `0`、`1`、`2` 或 `3`。degree 0 仅使用 DC color。

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m lic_mapping.trainer \
  --rosbag /home/DL/Projects/02_Thesis/03_Datasets/001_rosbags/Downtown1 \
  --calibration /home/DL/Projects/02_Thesis/03_Datasets/001_rosbags/Downtown1/cam_in_ex.txt \
  --output /home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_sh0.pt \
  --artifact-dir /home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_sh0_artifacts \
  --device cuda \
  --resize-width 640 \
  --resize-height 512 \
  --iterations 30 \
  --keyframe-every 8 \
  --sh-degree 0 \
  --max-gaussians 250000 \
  --prune-every 5 \
  --prune-opacity 0.01 \
  --spnet-weights /home/DL/Projects/02_Thesis/00_Baselines/SAGE-models/Large_300.pth \
  --lpips-backbone /home/DL/Projects/02_Thesis/00_Baselines/SAGE-models/alexnet-owt-7be5be79.pth
```

将命令中的 `--sh-degree 0` 改为 `1`、`2` 或 `3`，并同步修改
`--output` 和 `--artifact-dir` 路径即可。

## 4. 批量比较 SH degree 0/1/2/3

每个 degree 使用独立 checkpoint 和 artifact 目录，避免互相覆盖。

```bash
for degree in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 \
  python -m lic_mapping.trainer \
    --rosbag /home/DL/Projects/02_Thesis/03_Datasets/001_rosbags/Downtown1 \
    --calibration /home/DL/Projects/02_Thesis/03_Datasets/001_rosbags/Downtown1/cam_in_ex.txt \
    --output /home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_sh${degree}.pt \
    --artifact-dir /home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_sh${degree}_artifacts \
    --device cuda \
    --resize-width 640 \
    --resize-height 512 \
    --iterations 30 \
    --keyframe-every 8 \
    --sh-degree ${degree} \
    --max-gaussians 250000 \
    --prune-every 5 \
    --prune-opacity 0.01 \
    --spnet-weights /home/DL/Projects/02_Thesis/00_Baselines/SAGE-models/Large_300.pth \
    --lpips-backbone /home/DL/Projects/02_Thesis/00_Baselines/SAGE-models/alexnet-owt-7be5be79.pth
done
```

## 5. 快速冒烟比较

先用少量帧确认四个 degree 都能跑通，再进行完整训练：

```bash
for degree in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 \
  python -m lic_mapping.trainer \
    --rosbag /home/DL/Projects/02_Thesis/03_Datasets/001_rosbags/Downtown1 \
    --calibration /home/DL/Projects/02_Thesis/03_Datasets/001_rosbags/Downtown1/cam_in_ex.txt \
    --output /home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_smoke_sh${degree}.pt \
    --artifact-dir /home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_smoke_sh${degree}_artifacts \
    --device cuda \
    --resize-width 160 \
    --resize-height 128 \
    --frame-limit 3 \
    --iterations 1 \
    --keyframe-every 1 \
    --sh-degree ${degree} \
    --prune-every 0
done
```

## 6. 查看评估结果

每个实验的最终指标位于对应 artifact 目录的 `metrics.json`：

```bash
for degree in 0 1 2 3; do
  echo "SH degree ${degree}"
  python - <<PY
import json
from pathlib import Path

path = Path("/home/DL/Projects/02_Thesis/outputs/LIC_mapping/Downtown1_sh${degree}_artifacts/metrics.json")
report = json.loads(path.read_text())
print(report["aggregate"])
PY
done
```

重点比较 `psnr`、`ssim`、`lpips`、`depth_mae_m` 和 `alpha_mean`，同时检查
`renders/rgb`、`renders/alpha` 与 `renders/error` 中的可视化结果。
