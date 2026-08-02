from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lic_mapping.gaussians import GaussianMap
from lic_mapping.optimizers import SparseGaussianAdam
from lic_mapping.rosbag import BagFrame, CameraIntrinsics


def _fixed_frames() -> tuple[BagFrame, ...]:
    payload = json.loads((Path(__file__).parent / "fixtures" / "fixed_mapping_frames.json").read_text())
    intrinsics = CameraIntrinsics(**payload["intrinsics"])
    return tuple(
        BagFrame(
            item["index"],
            item["timestamp_ns"],
            np.asarray(item["rgb"], dtype=np.float32),
            np.asarray(item["depth_m"], dtype=np.float32),
            intrinsics,
            np.asarray(item["world_from_camera"], dtype=np.float64),
            np.asarray(item["points_world"], dtype=np.float32),
            np.asarray(item["point_colors"], dtype=np.float32),
        )
        for item in payload["frames"]
    )


def test_fixed_frame_contract_contains_the_mapping_inputs() -> None:
    frames = _fixed_frames()

    assert len(frames) == 2
    assert frames[0].rgb.shape == (4, 4, 3)
    assert frames[0].depth_m.shape == (4, 4)
    assert frames[0].points_world.shape == (3, 3)
    assert frames[0].point_colors.shape == (3, 3)
    assert np.array_equal(frames[0].world_from_camera, np.eye(4))
    assert np.isclose(frames[1].world_from_camera[0, 3], 0.1)
    assert np.isclose(frames[1].point_depths_m[0], 2.1)


def test_mapping_core_replays_the_same_frame_contract() -> None:
    first, second = _fixed_frames()
    model = GaussianMap.from_frame(first, device="cpu")
    optimizer = SparseGaussianAdam(model.parameters(), eps=1e-15)
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    optimizer.step()

    added = model.append_frame(second, optimizer=optimizer, alpha_gate=False, pixel_dedup=False)

    assert added == len(second.points_world)
    assert torch.allclose(model.means3d[-added:].cpu(), torch.from_numpy(second.points_world))
    assert torch.allclose(model.dc[-added:, 0].cpu(), (torch.from_numpy(second.point_colors) - 0.5) / 0.28209479177387814)
