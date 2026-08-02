from __future__ import annotations

import numpy as np
import torch

from lic_mapping.gaussians import GaussianMap
from lic_mapping.rosbag import BagFrame, CameraIntrinsics, PoseTrack, SOURCE_CENTER, _decode_cloud


def _frame(index: int, points: np.ndarray) -> BagFrame:
    intrinsics = CameraIntrinsics(8, 8, 8.0, 8.0, 3.5, 3.5)
    rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
    depth = np.zeros((8, 8), dtype=np.float32)
    colors = np.linspace(0.1, 0.9, len(points) * 3, dtype=np.float32).reshape(-1, 3)
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


def test_pose_track_interpolates_translation_and_rotation() -> None:
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    second[0, 3] = 2.0
    pose = PoseTrack([(0, first), (2_000_000_000, second)]).interpolate(
        1_000_000_000,
        max_dt_ns=1_000_000_000,
    )
    assert np.isclose(pose[0, 3], 1.0)
    assert np.allclose(pose[:3, :3], np.eye(3))


def test_gaussian_append_preserves_adam_state() -> None:
    initial = _frame(0, np.asarray([[0, 0, 2], [0.2, 0, 2], [0, 0.2, 2]], dtype=np.float32))
    model = GaussianMap.from_frame(initial, device="cpu", voxel_size=0.05)
    assert model.source_types.shape == (3,)
    assert torch.all(model.source_types == int(SOURCE_CENTER))
    assert torch.allclose(model.scales, model.scales[:, :1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    (sum(parameter.square().mean() for parameter in model.parameters())).backward()
    optimizer.step()

    next_frame = _frame(1, np.asarray([[0, 0, 2], [0.4, 0, 2], [0, 0.4, 2]], dtype=np.float32))
    added = model.append_frame(next_frame, optimizer=optimizer, alpha_gate=False)

    assert added == 2
    assert model.count == 5
    assert model.source_types.shape == (5,)
    assert torch.all(model.source_confidences == 1)
    assert optimizer.state[model.means3d]["exp_avg"].shape == (5, 3)
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    optimizer.step()


def test_gaussian_prune_migrates_rows_and_keeps_rasterizer_floor() -> None:
    frame = _frame(0, np.asarray([[0, 0, 2], [0.2, 0, 2], [0, 0.2, 2]], dtype=np.float32))
    model = GaussianMap.from_frame(frame, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.opacity_logits.data.fill_(-10.0)
    model.opacity_logits.data[:2] = 10.0

    removed = model.prune_low_opacity(optimizer=optimizer, threshold=0.5)

    assert removed == 1
    assert model.count == 2
    assert all(group["params"][0] is getattr(model, name) for group, name in zip(optimizer.param_groups, model.PARAMETER_NAMES))


def test_decode_uint32_rgb_field() -> None:
    packed = np.asarray([0x123456, 0xABCDEF], dtype=np.uint32)
    payload = packed.tobytes()
    message = {
        "big_endian": False,
        "width": 2,
        "height": 1,
        "point_step": 4,
        "fields": [("rgb", 0, 6, 1)],
        "data": payload,
    }
    points, colors = _decode_cloud({
        **message,
        "fields": [("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1), ("rgb", 12, 6, 1)],
        "point_step": 16,
        "data": np.asarray(
            [(1.0, 2.0, 3.0, packed[0]), (2.0, 3.0, 4.0, packed[1])],
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")],
        ).tobytes(),
    })
    assert points.shape == (2, 3)
    assert np.allclose(colors[0], np.asarray([0x12, 0x34, 0x56]) / 255.0)
