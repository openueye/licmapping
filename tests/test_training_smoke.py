from __future__ import annotations

import numpy as np
import pytest
import torch

from lic_mapping import evaluation
from lic_mapping.evaluation import evaluate_final_map
from lic_mapping.gaussians import GaussianMap
from lic_mapping.rosbag import BagFrame, CameraIntrinsics
from lic_mapping.trainer import LICMappingTrainer, TrainingConfig, _keyframe_view


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
def test_fixed_pose_training_loop_accumulates_and_updates(capsys: pytest.CaptureFixture[str]) -> None:
    frames = [
        _frame(0, np.asarray([[-0.2, 0, 2], [0.2, 0, 2], [0, -0.2, 2]], dtype=np.float32)),
        *(
            _frame(
                index,
                np.asarray([[-0.2, 0, 2], [0.2, 0, 2], [0, 0.2, 2], [0.3, 0.2, 2]], dtype=np.float32),
            )
            for index in range(1, 20)
        ),
    ]
    trainer = LICMappingTrainer(
        TrainingConfig(
            iterations_per_frame=1,
            keyframe_every=1,
            replay_keyframes=0,
            max_initial_points=16,
            max_new_points_per_frame=16,
        ),
        device="cuda",
    )
    model, report = trainer.fit(frames)

    assert model.count > 3
    assert report["frames"] == 20
    assert report["test_views"] == 0
    assert len(report["history"]) == 20
    assert any(record["added"] > 0 for record in report["history"][1:])
    assert report["renderer_alignment"] == "native_reference"
    assert all(np.isfinite(record["loss"]) for record in report["history"])
    output = capsys.readouterr().out
    assert "LIC keyframe 20: frame=19" in output
    assert "optimized_views=" in output
    assert "LIC keyframe 1:" not in output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_pose_training_retains_non_keyframe_test_views() -> None:
    frames = [
        _frame(index, np.asarray([[-0.2, 0, 2], [0.2, 0, 2], [0, 0.2, 2]], dtype=np.float32))
        for index in range(4)
    ]
    trainer = LICMappingTrainer(
        TrainingConfig(iterations_per_frame=1, keyframe_every=2, max_initial_points=16, max_new_points_per_frame=16),
        device="cuda",
    )
    _model, report = trainer.fit(frames)

    assert report["keyframes"] == 2
    assert report["test_views"] == 2
    assert [view.index for view in trainer.last_keyframes] == [1, 3]
    assert [view.index for view in trainer.last_test_views] == [0, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_final_evaluation_keeps_train_and_test_views_separate(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _MetricStub:
        identity = {"kind": "test"}

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, *_args, **_kwargs) -> dict[str, float]:
            return {"psnr": 1.0, "ssim": 0.5, "lpips": 0.25}

    monkeypatch.setattr(evaluation, "SAGEImageMetricEvaluator", _MetricStub)
    train = _frame(10, np.asarray([[-0.2, 0, 2], [0.2, 0, 2], [0, 0.2, 2]], dtype=np.float32))
    test = _frame(11, np.asarray([[-0.2, 0, 2], [0.2, 0, 2], [0, 0.2, 2]], dtype=np.float32))
    model = GaussianMap.from_frame(train, device="cuda")

    metrics = evaluate_final_map(
        model,
        [_keyframe_view(train)],
        tmp_path,
        test_views=[_keyframe_view(test)],
    )

    assert metrics["aggregate"]["train"]["keyframes"] == 1
    assert metrics["aggregate"]["test"]["keyframes"] == 1
    assert (tmp_path / "train" / "renders" / "rgb" / "000010.png").is_file()
    assert (tmp_path / "test" / "renders" / "rgb" / "000011.png").is_file()
