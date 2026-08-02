from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import GaussianMap
from .rosbag import BagFrame, RosbagReader


@dataclass(frozen=True)
class TrainingConfig:
    iterations_per_frame: int = 30
    keyframe_every: int = 5
    replay_keyframes: int = 2
    max_initial_points: int = 100_000
    max_new_points_per_frame: int = 5_000
    voxel_size: float = 0.05
    initial_opacity: float = 0.5
    scale_clamp_min: float = 1e-4
    scale_anisotropy: tuple[float, float, float] = (0.95, 1.05, 1.20)
    learning_rate_means: float = 1.6e-4
    learning_rate_dc: float = 2.5e-3
    learning_rate_opacity: float = 5.0e-2
    learning_rate_scales: float = 1.0e-3
    learning_rate_rotations: float = 1.0e-3
    rgb_weight: float = 1.0
    depth_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.iterations_per_frame < 1 or self.keyframe_every < 1 or self.replay_keyframes < 0:
            raise ValueError("training iteration and keyframe settings are invalid")
        if self.max_initial_points < 2 or self.max_new_points_per_frame < 1:
            raise ValueError("point budgets are invalid")
        if self.voxel_size <= 0 or self.scale_clamp_min <= 0:
            raise ValueError("voxel_size and scale_clamp_min must be positive")
        if len(self.scale_anisotropy) != 3 or any(value <= 0 or not np.isfinite(value) for value in self.scale_anisotropy):
            raise ValueError("scale_anisotropy must contain three positive finite values")
        if not 0 < self.initial_opacity < 1:
            raise ValueError("initial_opacity must be within (0, 1)")
        rates = (
            self.learning_rate_means,
            self.learning_rate_dc,
            self.learning_rate_opacity,
            self.learning_rate_scales,
            self.learning_rate_rotations,
        )
        if any(rate <= 0 for rate in rates) or self.rgb_weight < 0 or self.depth_weight < 0:
            raise ValueError("learning rates and loss weights must be non-negative as appropriate")


class LICMappingTrainer:
    """Sequential fixed-pose mapper using the LIC rasterizer as its renderer."""

    def __init__(self, config: TrainingConfig, *, device: torch.device | str = "cuda") -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("LICMappingTrainer requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

    def fit(self, frames: list[BagFrame]) -> tuple[GaussianMap, dict[str, object]]:
        if not frames:
            raise ValueError("At least one ROSBAG frame is required")
        torch.manual_seed(0)
        first = _limit_frame(frames[0], self.config.max_initial_points)
        model = GaussianMap.from_frame(
            first,
            device=self.device,
            initial_opacity=self.config.initial_opacity,
            scale_clamp_min=self.config.scale_clamp_min,
            scale_anisotropy=self.config.scale_anisotropy,
            voxel_size=self.config.voxel_size,
        )
        optimizer = self._optimizer(model)
        keyframes: list[BagFrame] = [first]
        records: list[dict[str, object]] = []
        for step, frame in enumerate(frames):
            if step != 0:
                added = model.append_frame(
                    frame,
                    optimizer=optimizer,
                    max_points=self.config.max_new_points_per_frame,
                    scale_clamp_min=self.config.scale_clamp_min,
                    scale_anisotropy=self.config.scale_anisotropy,
                )
            else:
                added = 0
            if step % self.config.keyframe_every == 0 and step != 0:
                keyframes.append(frame)
            loss_record = self._optimize_frame(model, optimizer, frame, keyframes)
            loss_record["frame_index"] = frame.index
            loss_record["gaussian_count"] = model.count
            loss_record["added"] = added
            records.append(loss_record)
        report = {
            "frames": len(frames),
            "gaussian_count": model.count,
            "training": asdict(self.config),
            "history": records,
            "pose_optimization": "disabled",
            "renderer": "lic_mapping._C",
        }
        return model, report

    def _optimizer(self, model: GaussianMap) -> torch.optim.Optimizer:
        return torch.optim.Adam([
            {"params": [model.means3d], "lr": self.config.learning_rate_means},
            {"params": [model.dc], "lr": self.config.learning_rate_dc},
            {"params": [model.opacity_logits], "lr": self.config.learning_rate_opacity},
            {"params": [model.log_scales], "lr": self.config.learning_rate_scales},
            {"params": [model.rotations], "lr": self.config.learning_rate_rotations},
        ], eps=1e-15)

    def _optimize_frame(self, model: GaussianMap, optimizer: torch.optim.Optimizer, current: BagFrame, keyframes: list[BagFrame]) -> dict[str, object]:
        historical = keyframes[-self.config.replay_keyframes :] if self.config.replay_keyframes else []
        views = [current] + [frame for frame in historical if frame.index != current.index]
        final_rgb = 0.0
        final_depth = 0.0
        for iteration in range(self.config.iterations_per_frame):
            selected = views[iteration % len(views)]
            optimizer.zero_grad(set_to_none=True)
            output = model.render(selected.lic_camera())
            target_rgb = torch.from_numpy(selected.rgb).permute(2, 0, 1).contiguous().to(self.device)
            rgb_loss = F.l1_loss(output.rgb, target_rgb)
            target_depth = torch.from_numpy(selected.depth_m).to(self.device)
            valid_depth = (target_depth > 0) & (output.depth > 0)
            if bool(valid_depth.any()):
                depth_loss = F.smooth_l1_loss(output.depth[valid_depth], target_depth[valid_depth])
            else:
                depth_loss = output.depth.sum() * 0.0
            loss = self.config.rgb_weight * rgb_loss + self.config.depth_weight * depth_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite LIC mapping loss at frame {selected.index}")
            loss.backward()
            optimizer.step()
            final_rgb = float(rgb_loss.detach())
            final_depth = float(depth_loss.detach())
        return {"rgb_loss": final_rgb, "depth_loss": final_depth, "loss": self.config.rgb_weight * final_rgb + self.config.depth_weight * final_depth}


def _limit_frame(frame: BagFrame, limit: int) -> BagFrame:
    if len(frame.points_world) <= limit:
        return frame
    indices = np.linspace(0, len(frame.points_world) - 1, limit, dtype=np.int64)
    return BagFrame(
        frame.index,
        frame.timestamp_ns,
        frame.rgb,
        frame.depth_m,
        frame.intrinsics,
        frame.world_from_camera,
        frame.points_world[indices],
        frame.point_colors[indices],
        center_depth_m=frame.center_depth_m,
        fused5_depth_m=frame.fused5_depth_m,
        source_types=frame.source_types,
        source_confidences=frame.source_confidences,
        point_source_types=frame.point_source_types[indices],
        point_source_confidences=frame.point_source_confidences[indices],
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a fixed-pose LIC Gaussian map from an Odin ROSBAG")
    parser.add_argument("--rosbag", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--keyframe-every", type=int, default=5)
    parser.add_argument("--max-new-points", type=int, default=5_000)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    args = parser.parse_args(argv)
    if (args.resize_width is None) != (args.resize_height is None):
        parser.error("--resize-width and --resize-height must be provided together")
    resize = None if args.resize_width is None else (args.resize_width, args.resize_height)
    reader = RosbagReader(args.rosbag, args.calibration, resize=resize)
    frames = reader.frames(limit=args.frame_limit)
    config = TrainingConfig(
        iterations_per_frame=args.iterations,
        keyframe_every=args.keyframe_every,
        max_new_points_per_frame=args.max_new_points,
    )
    model, report = LICMappingTrainer(config, device=args.device).fit(frames)
    report["skipped_pose_frames"] = reader.skipped_pose_frames
    report["rejected_source_frames"] = reader.rejected_source_frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "report": report}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "gaussian_count": model.count,
        "frames": len(frames),
        "skipped_pose_frames": reader.skipped_pose_frames,
        "rejected_source_frames": reader.rejected_source_frames,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
