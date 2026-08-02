from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import GaussianMap
from .optimizers import SparseGaussianAdam
from .rosbag import BagFrame, RosbagReader, SOURCE_SPNET
from .spnet import DepthCompleter, complete_keyframe_points


@dataclass(frozen=True)
class TrainingConfig:
    iterations_per_frame: int = 30
    keyframe_every: int = 8
    replay_keyframes: int = 0  # retained for checkpoint/config compatibility; LIC2 uses all train keyframes
    max_initial_points: int | None = None
    max_new_points_per_frame: int | None = None
    initial_opacity: float = 0.1
    growth_opacity: float = 0.1
    scale_clamp_min: float = 1e-4
    scale_multiplier: float = 2.0
    scale_anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sh_degree: int = 3
    max_gaussians: int = 250_000
    prune_opacity_threshold: float = 0.01
    prune_every_n_keyframes: int = 5
    learning_rate_means: float = 1.6e-4
    learning_rate_dc: float = 5.0e-3
    learning_rate_opacity: float = 5.0e-2
    learning_rate_scales: float = 1.5e-2
    learning_rate_rotations: float = 1.0e-3
    rgb_weight: float = 1.0
    lambda_dssim: float = 0.2
    optimize_depth: bool = True
    depth_weight: float = 0.005
    iteration_decay: bool = True
    depth_completion: bool = False
    depth_completion_patch_size: int = 10
    depth_completion_max_depth_m: float = 20.0
    depth_completion_confidence: float = 0.4

    def __post_init__(self) -> None:
        if self.iterations_per_frame < 1 or self.keyframe_every < 1 or self.replay_keyframes < 0:
            raise ValueError("training iteration and keyframe settings are invalid")
        if self.max_initial_points is not None and self.max_initial_points < 2:
            raise ValueError("max_initial_points must be at least two when specified")
        if self.max_new_points_per_frame is not None and self.max_new_points_per_frame < 1:
            raise ValueError("max_new_points_per_frame must be positive when specified")
        if self.scale_clamp_min <= 0:
            raise ValueError("scale_clamp_min must be positive")
        if len(self.scale_anisotropy) != 3 or any(value <= 0 or not np.isfinite(value) for value in self.scale_anisotropy):
            raise ValueError("scale_anisotropy must contain three positive finite values")
        if not 0 < self.initial_opacity < 1 or not 0 < self.growth_opacity < 1:
            raise ValueError("initial_opacity and growth_opacity must be within (0, 1)")
        if self.scale_multiplier <= 0 or not np.isfinite(self.scale_multiplier):
            raise ValueError("scale_multiplier must be positive and finite")
        if self.sh_degree < 0 or self.sh_degree > 3:
            raise ValueError("sh_degree must be within [0, 3]")
        if self.max_gaussians < 2:
            raise ValueError("max_gaussians must be at least two")
        if self.prune_every_n_keyframes < 0 or self.prune_opacity_threshold < 0:
            raise ValueError("pruning settings are invalid")
        if self.depth_completion_patch_size < 1 or self.depth_completion_max_depth_m <= 0:
            raise ValueError("depth completion settings must be positive")
        if not 0 <= self.depth_completion_confidence <= 1:
            raise ValueError("depth_completion_confidence must be within [0, 1]")
        if not 0 <= self.lambda_dssim <= 1:
            raise ValueError("lambda_dssim must be within [0, 1]")
        rates = (
            self.learning_rate_means,
            self.learning_rate_dc,
            self.learning_rate_opacity,
            self.learning_rate_scales,
            self.learning_rate_rotations,
        )
        if any(rate <= 0 for rate in rates) or self.rgb_weight < 0 or self.depth_weight < 0:
            raise ValueError("learning rates and loss weights must be non-negative as appropriate")


@dataclass
class _PointAccumulator:
    points: list[np.ndarray]
    colors: list[np.ndarray]
    depths: list[np.ndarray]
    source_types: list[np.ndarray]
    source_confidences: list[np.ndarray]

    @classmethod
    def empty(cls) -> "_PointAccumulator":
        return cls([], [], [], [], [])

    def add(self, frame: BagFrame) -> None:
        self.points.append(frame.points_world)
        self.colors.append(frame.point_colors)
        self.depths.append(frame.point_depths_m)
        self.source_types.append(frame.point_source_types)
        self.source_confidences.append(frame.point_source_confidences)

    def frame(self, reference: BagFrame) -> BagFrame:
        if not self.points:
            raise ValueError("Cannot materialize an empty point accumulator")
        return BagFrame(
            reference.index,
            reference.timestamp_ns,
            reference.rgb,
            reference.depth_m,
            reference.intrinsics,
            reference.world_from_camera,
            np.concatenate(self.points, axis=0).astype(np.float32, copy=False),
            np.concatenate(self.colors, axis=0).astype(np.float32, copy=False),
            point_depths_m=np.concatenate(self.depths, axis=0).astype(np.float32, copy=False),
            center_depth_m=reference.center_depth_m,
            fused5_depth_m=reference.fused5_depth_m,
            source_types=reference.source_types,
            source_confidences=reference.source_confidences,
            point_source_types=np.concatenate(self.source_types, axis=0).astype(np.uint8, copy=False),
            point_source_confidences=np.concatenate(self.source_confidences, axis=0).astype(np.float32, copy=False),
        )

    def clear(self) -> None:
        self.points.clear()
        self.colors.clear()
        self.depths.clear()
        self.source_types.clear()
        self.source_confidences.clear()

    def add_completed(self, points: np.ndarray, colors: np.ndarray, depths: np.ndarray, *, confidence: float) -> None:
        if not len(points):
            return
        self.points.append(np.asarray(points, dtype=np.float32))
        self.colors.append(np.asarray(colors, dtype=np.float32))
        self.depths.append(np.asarray(depths, dtype=np.float32))
        self.source_types.append(np.full(len(points), int(SOURCE_SPNET), dtype=np.uint8))
        self.source_confidences.append(np.full(len(points), confidence, dtype=np.float32))


@dataclass(frozen=True)
class _KeyframeView:
    index: int
    rgb: np.ndarray
    depth_m: np.ndarray
    world_from_camera: np.ndarray
    camera: object


class LICMappingTrainer:
    """Sequential fixed-pose mapper using SAGE's CUDA rasterizer."""

    def __init__(self, config: TrainingConfig, *, device: torch.device | str = "cuda", depth_completer: DepthCompleter | None = None) -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("LICMappingTrainer requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if config.depth_completion and depth_completer is None:
            raise ValueError("depth_completion=True requires a DepthCompleter")
        self.depth_completer = depth_completer
        self.last_keyframes: list[_KeyframeView] = []
        self.last_depth_completion: list[dict[str, object]] = []

    def fit(self, frames: Iterable[BagFrame]) -> tuple[GaussianMap, dict[str, object]]:
        torch.manual_seed(0)
        random.seed(0)
        self._rng = random.Random(0)
        accumulator = _PointAccumulator.empty()
        model: GaussianMap | None = None
        optimizer: SparseGaussianAdam | None = None
        keyframes: list[_KeyframeView] = []
        records: list[dict[str, object]] = []
        accepted_frames = 0
        keyframe_count = 0
        for frame in frames:
            accepted_frames += 1
            accumulator.add(frame)
            if accepted_frames % self.config.keyframe_every != 0:
                continue
            completion_record: dict[str, object] = {"enabled": self.config.depth_completion, "added": 0}
            if self.config.depth_completion:
                assert self.depth_completer is not None
                completion = complete_keyframe_points(
                    frame,
                    self.depth_completer,
                    patch_size=self.config.depth_completion_patch_size,
                    max_depth_m=self.config.depth_completion_max_depth_m,
                )
                accumulator.add_completed(
                    completion.points_world,
                    completion.colors,
                    completion.depths_m,
                    confidence=self.config.depth_completion_confidence,
                )
                completion_record.update({
                    "added": int(len(completion.points_world)),
                    "candidate_count": completion.candidate_count,
                    "mean_known_bias_m": completion.mean_known_bias_m,
                })
                self.last_depth_completion.append({"frame_index": frame.index, **completion_record})
            accumulated = accumulator.frame(frame)
            added = 0
            clear_accumulator = True
            if model is None:
                initial = _limit_frame(accumulated, self.config.max_initial_points)
                model = GaussianMap.from_frame(
                    initial,
                    device=self.device,
                    initial_opacity=self.config.initial_opacity,
                    scale_clamp_min=self.config.scale_clamp_min,
                    scale_anisotropy=self.config.scale_anisotropy,
                    scale_multiplier=self.config.scale_multiplier,
                    sh_degree=self.config.sh_degree,
                )
                optimizer = self._optimizer(model)
            else:
                if model.count < self.config.max_gaussians:
                    remaining = self.config.max_gaussians - model.count
                    budget = self.config.max_new_points_per_frame
                    if budget is not None:
                        budget = min(budget, remaining)
                    assert optimizer is not None
                    added = model.append_frame(
                        accumulated,
                        optimizer=optimizer,
                        max_points=budget,
                        scale_clamp_min=self.config.scale_clamp_min,
                        scale_anisotropy=self.config.scale_anisotropy,
                        scale_multiplier=self.config.scale_multiplier,
                        growth_opacity=self.config.growth_opacity,
                        alpha_gate=True,
                        pixel_dedup=True,
                    )
                else:
                    # LIC2 keeps Dataset::pointcloud_ when the hard cap blocks
                    # extend(), so pruning can release space for the next keyframe.
                    clear_accumulator = False
            keyframes.append(_keyframe_view(frame))
            keyframe_count += 1
            assert model is not None and optimizer is not None
            loss_record = self._optimize_keyframe(model, optimizer, keyframes)
            pruned = 0
            if self.config.prune_every_n_keyframes and keyframe_count % self.config.prune_every_n_keyframes == 0:
                pruned = model.prune_low_opacity(
                    optimizer=optimizer,
                    threshold=self.config.prune_opacity_threshold,
                )
            loss_record.update({
                "frame_index": frame.index,
                "gaussian_count": model.count,
                "added": added,
                "pruned": pruned,
                "depth_completion": completion_record,
            })
            records.append(loss_record)
            if clear_accumulator:
                accumulator.clear()
            del accumulated
        if model is None or optimizer is None:
            raise ValueError("No keyframe was available for Gaussian initialization")
        spnet_identity = None
        if self.depth_completer is not None:
            spnet_identity = {
                name: str(getattr(self.depth_completer, name))
                for name in (
                    "model_id",
                    "source_id",
                    "source_commit",
                    "source_tree_sha256",
                    "weights_path",
                    "weights_sha256",
                )
                if hasattr(self.depth_completer, name)
            }
        report = {
            "frames": accepted_frames,
            "keyframes": keyframe_count,
            "gaussian_count": model.count,
            "training": asdict(self.config),
            "history": records,
            "pose_optimization": "disabled",
            "renderer": "sage.diff_gaussian_rasterization",
            "depth_completion": {
                "enabled": self.config.depth_completion,
                "backend": type(self.depth_completer).__name__ if self.depth_completer is not None else None,
                "identity": spnet_identity,
                "patch_size": self.config.depth_completion_patch_size,
                "max_depth_m": self.config.depth_completion_max_depth_m,
                "keyframes": self.last_depth_completion,
            },
        }
        self.last_keyframes = keyframes
        return model, report

    def _optimizer(self, model: GaussianMap) -> SparseGaussianAdam:
        return SparseGaussianAdam([
            {"params": [model.means3d], "lr": self.config.learning_rate_means},
            {"params": [model.dc], "lr": self.config.learning_rate_dc},
            {"params": [model.sh_rest], "lr": self.config.learning_rate_dc / 20.0},
            {"params": [model.opacity_logits], "lr": self.config.learning_rate_opacity},
            {"params": [model.log_scales], "lr": self.config.learning_rate_scales},
            {"params": [model.rotations], "lr": self.config.learning_rate_rotations},
        ], eps=1e-15)

    def _optimize_keyframe(self, model: GaussianMap, optimizer: SparseGaussianAdam, keyframes: list[_KeyframeView]) -> dict[str, object]:
        if not keyframes:
            raise ValueError("At least one keyframe is required")
        selected = list(keyframes)
        max_views = self.config.iterations_per_frame
        distance = float(np.linalg.norm(keyframes[-1].world_from_camera[:3, 3] - keyframes[0].world_from_camera[:3, 3]))
        if self.config.iteration_decay and distance > 120.0:
            max_views = max(1, max_views // 2)
            split = len(keyframes) * 2 // 3
            half = max(1, max_views // 2)
            selected = self._rng.sample(keyframes[:split], min(half, split)) if split else []
            selected += self._rng.sample(keyframes[split:], min(half, len(keyframes) - split))
        elif len(selected) > max_views:
            selected = self._rng.sample(selected, max_views)
        else:
            self._rng.shuffle(selected)
        final_rgb = 0.0
        final_photo = 0.0
        final_depth = 0.0
        for frame in selected:
            optimizer.zero_grad(set_to_none=True)
            output = model.render(frame.camera)
            target_rgb = torch.from_numpy(frame.rgb).permute(2, 0, 1).contiguous().to(self.device)
            rendered_rgb = output.rgb
            rgb_loss = F.l1_loss(rendered_rgb, target_rgb)
            photo_loss = (1.0 - self.config.lambda_dssim) * rgb_loss + self.config.lambda_dssim * (1.0 - _ssim(rendered_rgb, target_rgb))
            target_depth = torch.from_numpy(frame.depth_m).to(self.device)
            valid_depth = (target_depth > 0) & (output.depth > 0)
            if self.config.optimize_depth and bool(valid_depth.any()):
                depth_loss = torch.abs(output.depth[valid_depth] - target_depth[valid_depth]).mean()
            else:
                depth_loss = output.depth.sum() * 0.0
            loss = self.config.rgb_weight * photo_loss + self.config.depth_weight * depth_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite LIC mapping loss at frame {frame.index}: "
                    f"rgb={float(rgb_loss.detach())}, photo={float(photo_loss.detach())}, "
                    f"depth={float(depth_loss.detach())}, "
                    f"raw_finite={bool(torch.isfinite(output.rgb).all())}, "
                    f"rendered_finite={bool(torch.isfinite(rendered_rgb).all())}, "
                    f"target_finite={bool(torch.isfinite(target_rgb).all())}, "
                    f"parameter_finite={all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())}"
                )
            loss.backward()
            optimizer.set_visibility(output.visible.detach())
            optimizer.step()
            final_rgb = float(rgb_loss.detach())
            final_photo = float(photo_loss.detach())
            final_depth = float(depth_loss.detach())
        return {
            "rgb_loss": final_rgb,
            "photo_loss": final_photo,
            "depth_loss": final_depth,
            "loss": self.config.rgb_weight * final_photo + self.config.depth_weight * final_depth,
            "optimized_views": len(selected),
        }


def _ssim(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if rendered.shape != target.shape or rendered.ndim != 3 or rendered.shape[0] != 3:
        raise ValueError("RGB tensors must share [3, H, W] shape")
    rendered = rendered.unsqueeze(0)
    target = target.unsqueeze(0)
    coordinates = torch.arange(11, dtype=rendered.dtype, device=rendered.device) - 5
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).view(1, 1, 11, 11).expand(3, 1, 11, 11).contiguous()
    mu_rendered = F.conv2d(rendered, window, padding=5, groups=3)
    mu_target = F.conv2d(target, window, padding=5, groups=3)
    sigma_rendered = F.conv2d(rendered * rendered, window, padding=5, groups=3) - mu_rendered.square()
    sigma_target = F.conv2d(target * target, window, padding=5, groups=3) - mu_target.square()
    sigma_cross = F.conv2d(rendered * target, window, padding=5, groups=3) - mu_rendered * mu_target
    score = (((2 * mu_rendered * mu_target + 0.01**2) * (2 * sigma_cross + 0.03**2)) /
             ((mu_rendered.square() + mu_target.square() + 0.01**2) * (sigma_rendered + sigma_target + 0.03**2))).mean()
    return score.clamp(0, 1)


def _keyframe_view(frame: BagFrame) -> _KeyframeView:
    return _KeyframeView(
        frame.index,
        frame.rgb,
        frame.depth_m,
        frame.world_from_camera,
        frame.lic_camera(),
    )


def _limit_frame(frame: BagFrame, limit: int | None) -> BagFrame:
    if limit is None or len(frame.points_world) <= limit:
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
        point_depths_m=frame.point_depths_m[indices],
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
    parser.add_argument("--keyframe-every", type=int, default=8)
    parser.add_argument("--sh-degree", type=int, choices=range(4), default=3,
                        help="Spherical-harmonics degree for Gaussian colors (0-3)")
    parser.add_argument("--max-new-points", type=int, default=None)
    parser.add_argument("--max-gaussians", type=int, default=250_000)
    parser.add_argument("--prune-every", type=int, default=5)
    parser.add_argument("--prune-opacity", type=float, default=0.01)
    parser.add_argument("--spnet-engine", type=Path, default=None)
    parser.add_argument("--spnet-torchscript", type=Path, default=None)
    parser.add_argument("--spnet-patch-size", type=int, default=10)
    parser.add_argument("--spnet-max-depth", type=float, default=20.0)
    parser.add_argument("--spnet-weights", type=Path, default=None)
    parser.add_argument("--spnet-source", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--lpips-backbone", type=Path, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    args = parser.parse_args(argv)
    if (args.resize_width is None) != (args.resize_height is None):
        parser.error("--resize-width and --resize-height must be provided together")
    resize = None if args.resize_width is None else (args.resize_width, args.resize_height)
    backends = [args.spnet_engine, args.spnet_torchscript, args.spnet_weights]
    if sum(value is not None for value in backends) > 1:
        parser.error("--spnet-engine, --spnet-torchscript, and --spnet-weights are mutually exclusive")
    reader = RosbagReader(args.rosbag, args.calibration, resize=resize)
    frames = reader.frames(limit=args.frame_limit)
    config = TrainingConfig(
        iterations_per_frame=args.iterations,
        keyframe_every=args.keyframe_every,
        sh_degree=args.sh_degree,
        max_new_points_per_frame=args.max_new_points,
        max_gaussians=args.max_gaussians,
        prune_every_n_keyframes=args.prune_every,
        prune_opacity_threshold=args.prune_opacity,
        depth_completion=any(value is not None for value in backends),
        depth_completion_patch_size=args.spnet_patch_size,
        depth_completion_max_depth_m=args.spnet_max_depth,
    )
    completer = None
    if args.spnet_engine is not None:
        from .spnet import TensorRTDepthCompleter
        if resize is None:
            parser.error("--spnet-engine requires --resize-width/--resize-height")
        completer = TensorRTDepthCompleter(args.spnet_engine, width=resize[0], height=resize[1], device=args.device)
    elif args.spnet_torchscript is not None:
        from .spnet import TorchScriptDepthCompleter
        completer = TorchScriptDepthCompleter(args.spnet_torchscript, device=args.device)
    elif args.spnet_weights is not None:
        from .spnet import SPNetDepthCompleter
        completer = SPNetDepthCompleter(
            args.spnet_weights,
            source_root=args.spnet_source,
            device=args.device,
        )
    trainer = LICMappingTrainer(config, device=args.device, depth_completer=completer)
    model, report = trainer.fit(frames)
    report["skipped_pose_frames"] = reader.skipped_pose_frames
    report["rejected_source_frames"] = reader.rejected_source_frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.artifact_dir or args.output.with_name(f"{args.output.stem}_artifacts")
    if not args.no_artifacts:
        from .evaluation import evaluate_final_map
        evaluation = evaluate_final_map(
            model,
            trainer.last_keyframes,
            artifact_dir,
            lpips_backbone=args.lpips_backbone,
        )
        report["evaluation"] = {
            "artifact_dir": str(artifact_dir),
            "aggregate": evaluation["aggregate"],
        }
    torch.save({"state_dict": model.state_dict(), "report": report}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "gaussian_count": model.count,
        "frames": report["frames"],
        "skipped_pose_frames": reader.skipped_pose_frames,
        "rejected_source_frames": reader.rejected_source_frames,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
