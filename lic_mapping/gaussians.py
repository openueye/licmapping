from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .rosbag import BagFrame, SOURCE_CENTER
from .rasterizer import LicRenderOutput, render


SH_C0 = 0.28209479177387814


class GaussianMap(nn.Module):
    """Trainable Gaussian state with append-only map accumulation."""

    PARAMETER_NAMES = ("means3d", "dc", "opacity_logits", "log_scales", "rotations")

    def __init__(
        self,
        means3d: torch.Tensor,
        dc: torch.Tensor,
        opacity_logits: torch.Tensor,
        log_scales: torch.Tensor,
        rotations: torch.Tensor,
        *,
        voxel_size: float = 0.05,
        source_types: torch.Tensor | None = None,
        source_confidences: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.means3d = nn.Parameter(means3d)
        self.dc = nn.Parameter(dc)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.log_scales = nn.Parameter(log_scales)
        self.rotations = nn.Parameter(rotations)
        count = int(means3d.shape[0])
        if source_types is None:
            source_types = torch.full((count,), int(SOURCE_CENTER), dtype=torch.uint8, device=means3d.device)
        if source_confidences is None:
            source_confidences = torch.ones(count, dtype=torch.float32, device=means3d.device)
        self.register_buffer("source_types", source_types.to(device=means3d.device, dtype=torch.uint8).contiguous())
        self.register_buffer("source_confidences", source_confidences.to(device=means3d.device, dtype=torch.float32).contiguous())
        self.voxel_size = float(voxel_size)
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive and finite")
        self._voxel_keys: set[tuple[int, int, int]] = set()
        self._validate()

    @classmethod
    def from_frame(
        cls,
        frame: BagFrame,
        *,
        device: torch.device | str,
        initial_opacity: float = 0.5,
        scale_clamp_min: float = 1e-4,
        scale_anisotropy: tuple[float, float, float] = (0.95, 1.05, 1.20),
        voxel_size: float = 0.05,
    ) -> "GaussianMap":
        if not 0.0 < initial_opacity < 1.0:
            raise ValueError("initial_opacity must be within (0, 1)")
        if scale_clamp_min <= 0 or not math.isfinite(scale_clamp_min):
            raise ValueError("scale_clamp_min must be positive and finite")
        anisotropy = np.asarray(scale_anisotropy, dtype=np.float32)
        if anisotropy.shape != (3,) or not np.isfinite(anisotropy).all() or (anisotropy <= 0).any():
            raise ValueError("scale_anisotropy must contain three positive finite values")
        if len(frame.points_world) < 2:
            raise ValueError("LIC rasterizer requires at least two initialized Gaussians")
        target = torch.device(device)
        points = torch.from_numpy(frame.points_world).to(target)
        colors = torch.from_numpy(frame.point_colors).to(target)
        camera_from_world = np.linalg.inv(frame.world_from_camera)
        points_h = np.concatenate((frame.points_world.astype(np.float64), np.ones((len(frame.points_world), 1))), axis=1)
        z = (camera_from_world @ points_h.T).T[:, 2]
        z = torch.from_numpy(np.maximum(z, 0.1).astype(np.float32)).to(target)
        focal = 0.5 * (frame.intrinsics.fx + frame.intrinsics.fy)
        base_scale = (z / focal).clamp_min(scale_clamp_min)
        log_scales = torch.log(base_scale[:, None] * torch.as_tensor(anisotropy, device=target))
        dc = ((colors - 0.5) / SH_C0).view(-1, 1, 3)
        opacity_logit = math.log(initial_opacity) - math.log1p(-initial_opacity)
        model = cls(
            points,
            dc,
            torch.full((len(points), 1), opacity_logit, dtype=torch.float32, device=target),
            log_scales,
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=target).repeat(len(points), 1),
            voxel_size=voxel_size,
            source_types=torch.from_numpy(frame.point_source_types).to(target),
            source_confidences=torch.from_numpy(frame.point_source_confidences).to(target),
        )
        model._voxel_keys = {
            tuple(np.floor(point / model.voxel_size).astype(np.int64).tolist())
            for point in frame.points_world
        }
        return model

    @property
    def count(self) -> int:
        return int(self.means3d.shape[0])

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def render(self, camera, *, debug: bool = False) -> LicRenderOutput:
        sh = torch.empty((0,), dtype=torch.float32, device=self.means3d.device)
        rotations = torch.nn.functional.normalize(self.rotations, dim=1)
        return render(
            self.means3d,
            self.dc,
            sh,
            self.opacities,
            self.scales,
            rotations,
            camera,
            sh_degree=0,
            debug=debug,
        )

    def append_frame(
        self,
        frame: BagFrame,
        *,
        optimizer: torch.optim.Optimizer,
        max_points: int | None = None,
        scale_clamp_min: float = 1e-4,
        scale_anisotropy: tuple[float, float, float] = (0.95, 1.05, 1.20),
    ) -> int:
        indices = self._new_point_indices(frame.points_world, max_points=max_points)
        if not len(indices):
            return 0
        candidate = BagFrame(
            frame.index,
            frame.timestamp_ns,
            frame.rgb,
            frame.depth_m,
            frame.intrinsics,
            frame.world_from_camera,
            frame.points_world[indices],
            frame.point_colors[indices],
            point_source_types=frame.point_source_types[indices],
            point_source_confidences=frame.point_source_confidences[indices],
        )
        additions = self._tensors_from_frame(
            candidate,
            scale_clamp_min=scale_clamp_min,
            scale_anisotropy=scale_anisotropy,
        )
        self._append_tensors(additions, optimizer)
        return len(indices)

    def _tensors_from_frame(self, frame: BagFrame, *, scale_clamp_min: float, scale_anisotropy: tuple[float, float, float]) -> tuple[torch.Tensor, ...]:
        target = self.means3d.device
        means = torch.from_numpy(frame.points_world).to(target)
        colors = torch.from_numpy(frame.point_colors).to(target)
        camera_from_world = np.linalg.inv(frame.world_from_camera)
        points_h = np.concatenate((frame.points_world.astype(np.float64), np.ones((len(frame.points_world), 1))), axis=1)
        z = (camera_from_world @ points_h.T).T[:, 2]
        z = torch.from_numpy(np.maximum(z, 0.1).astype(np.float32)).to(target)
        base_scale = (z / (0.5 * (frame.intrinsics.fx + frame.intrinsics.fy))).clamp_min(scale_clamp_min)
        scales = base_scale[:, None] * torch.as_tensor(scale_anisotropy, dtype=torch.float32, device=target)
        return (
            means,
            ((colors - 0.5) / SH_C0).view(-1, 1, 3),
            torch.full((len(means), 1), float(self.opacity_logits.detach()[0, 0]), device=target),
            torch.log(scales),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=target).repeat(len(means), 1),
            torch.from_numpy(frame.point_source_types).to(target),
            torch.from_numpy(frame.point_source_confidences).to(target),
        )

    def _new_point_indices(self, points: np.ndarray, *, max_points: int | None) -> np.ndarray:
        if max_points is not None and max_points < 1:
            raise ValueError("max_points must be positive")
        indices: list[int] = []
        for index, point in enumerate(np.asarray(points)):
            key = tuple(np.floor(point / self.voxel_size).astype(np.int64).tolist())
            if key in self._voxel_keys:
                continue
            self._voxel_keys.add(key)
            indices.append(index)
            if max_points is not None and len(indices) >= max_points:
                break
        return np.asarray(indices, dtype=np.int64)

    def _append_tensors(self, additions: tuple[torch.Tensor, ...], optimizer: torch.optim.Optimizer) -> None:
        old_count = self.count
        old_parameters = {name: getattr(self, name) for name in self.PARAMETER_NAMES}
        for name, addition in zip(self.PARAMETER_NAMES, additions[: len(self.PARAMETER_NAMES)]):
            old = old_parameters[name]
            new = nn.Parameter(torch.cat((old.detach(), addition.detach()), dim=0))
            self._replace_optimizer_parameter(optimizer, old, new, old_count)
            setattr(self, name, new)
        self.source_types = torch.cat((self.source_types, additions[5].detach().to(self.source_types.device)), dim=0)
        self.source_confidences = torch.cat((self.source_confidences, additions[6].detach().to(self.source_confidences.device)), dim=0)
        self._validate()

    @staticmethod
    def _replace_optimizer_parameter(optimizer: torch.optim.Optimizer, old: nn.Parameter, new: nn.Parameter, old_count: int) -> None:
        for group in optimizer.param_groups:
            group["params"] = [new if parameter is old else parameter for parameter in group["params"]]
        state = optimizer.state.pop(old, {})
        migrated = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == old_count:
                padding = torch.zeros((new.shape[0] - old_count, *value.shape[1:]), dtype=value.dtype, device=value.device)
                migrated[key] = torch.cat((value, padding), dim=0)
            else:
                migrated[key] = value
        optimizer.state[new] = migrated

    def _validate(self) -> None:
        count = self.means3d.shape[0]
        if count < 2:
            raise ValueError("GaussianMap requires at least two rows for the LIC reference kernel")
        if self.means3d.shape != (count, 3) or self.dc.shape != (count, 1, 3):
            raise ValueError("GaussianMap means3d/dc shapes are invalid")
        if self.opacity_logits.shape != (count, 1) or self.log_scales.shape != (count, 3) or self.rotations.shape != (count, 4):
            raise ValueError("GaussianMap parameter shapes are invalid")
        if self.source_types.shape != (count,) or self.source_types.dtype != torch.uint8:
            raise ValueError("GaussianMap source_types shape or dtype is invalid")
        if self.source_confidences.shape != (count,) or self.source_confidences.dtype != torch.float32:
            raise ValueError("GaussianMap source_confidences shape or dtype is invalid")
        if not all(torch.isfinite(parameter).all() for parameter in self.parameters()):
            raise ValueError("GaussianMap parameters must be finite")
        if not torch.isfinite(self.source_confidences).all() or (self.source_confidences < 0).any():
            raise ValueError("GaussianMap source_confidences must be finite and non-negative")
