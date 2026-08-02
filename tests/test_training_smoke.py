from __future__ import annotations

import numpy as np
import pytest
import torch

from lic_mapping.rosbag import BagFrame, CameraIntrinsics
from lic_mapping.trainer import LICMappingTrainer, TrainingConfig


def _frame(index: int, points: np.ndarray) -> BagFrame:
    intrinsics = CameraIntrinsics(16, 16, 12.0, 12.0, 7.5, 7.5)
    rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    depth = np.zeros((16, 16), dtype=np.float32)
    colors = np.asarray([
        [0.9, 0.2, 0.1],
        [0.1, 0.8, 0.2],
        [0.2, 0.3, 0.9],
        [0.8, 0.8, 0.2],
        [0.2, 0.8, 0.8],
    ], dtype=np.float32)[: len(points)]
    return BagFrame(
        index,
        index * 1_000_000_000,
        rgb,
        depth,
        intrinsics,
        np.eye(4, dtype=np.float64),
        points.astype(np.float32),
        colors,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_pose_training_loop_accumulates_and_updates() -> None:
    frames = [
        _frame(0, np.asarray([[-0.2, 0, 2], [0.2, 0, 2], [0, -0.2, 2]], dtype=np.float32)),
        _frame(1, np.asarray([[-0.2, 0, 2], [0.2, 0, 2], [0, 0.2, 2], [0.3, 0.2, 2]], dtype=np.float32)),
    ]
    trainer = LICMappingTrainer(
        TrainingConfig(
            iterations_per_frame=1,
            keyframe_every=1,
            replay_keyframes=0,
            max_initial_points=16,
            max_new_points_per_frame=16,
            voxel_size=0.05,
        ),
        device="cuda",
    )
    model, report = trainer.fit(frames)

    assert model.count == 5
    assert report["frames"] == 2
    assert len(report["history"]) == 2
    assert all(np.isfinite(record["loss"]) for record in report["history"])
