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

    PARAMETER_NAMES = ("means3d", "dc", "sh_rest", "opacity_logits", "log_scales", "rotations")

    def __init__(
        self,
        means3d: torch.Tensor,
        dc: torch.Tensor,
        opacity_logits: torch.Tensor,
        log_scales: torch.Tensor,
        rotations: torch.Tensor,
        *,
        sh_rest: torch.Tensor | None = None,
        sh_degree: int = 3,
        source_types: torch.Tensor | None = None,
        source_confidences: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.means3d = nn.Parameter(means3d)
        self.dc = nn.Parameter(dc)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.log_scales = nn.Parameter(log_scales)
        self.rotations = nn.Parameter(rotations)
        if sh_degree < 0 or sh_degree > 3:
            raise ValueError("sh_degree must be within [0, 3]")
        self.sh_degree = int(sh_degree)
        count = int(means3d.shape[0])
        expected_sh = (self.sh_degree + 1) ** 2 - 1
        if sh_rest is None:
            sh_rest = torch.zeros((count, expected_sh, 3), dtype=torch.float32, device=means3d.device)
        if sh_rest.shape != (count, expected_sh, 3):
            raise ValueError("sh_rest shape is inconsistent with sh_degree")
        self.sh_rest = nn.Parameter(sh_rest)
        if source_types is None:
            source_types = torch.full((count,), int(SOURCE_CENTER), dtype=torch.uint8, device=means3d.device)
        if source_confidences is None:
            source_confidences = torch.ones(count, dtype=torch.float32, device=means3d.device)
        self.register_buffer("source_types", source_types.to(device=means3d.device, dtype=torch.uint8).contiguous())
        self.register_buffer("source_confidences", source_confidences.to(device=means3d.device, dtype=torch.float32).contiguous())
        self._validate()

    @classmethod
    def from_frame(
        cls,
        frame: BagFrame,
        *,
        device: torch.device | str,
        initial_opacity: float = 0.1,
        scale_clamp_min: float = 1e-4,
        scale_anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
        scale_multiplier: float = 2.0,
        sh_degree: int = 3,
    ) -> "GaussianMap":
        if not 0.0 < initial_opacity < 1.0:
            raise ValueError("initial_opacity must be within (0, 1)")
        if scale_clamp_min <= 0 or not math.isfinite(scale_clamp_min):
            raise ValueError("scale_clamp_min must be positive and finite")
        if scale_multiplier <= 0 or not math.isfinite(scale_multiplier):
            raise ValueError("scale_multiplier must be positive and finite")
        if sh_degree < 0 or sh_degree > 3:
            raise ValueError("sh_degree must be within [0, 3]")
        anisotropy = np.asarray(scale_anisotropy, dtype=np.float32)
        if anisotropy.shape != (3,) or not np.isfinite(anisotropy).all() or (anisotropy <= 0).any():
            raise ValueError("scale_anisotropy must contain three positive finite values")
        if len(frame.points_world) < 2:
            raise ValueError("LIC rasterizer requires at least two initialized Gaussians")
        target = torch.device(device)
        points = torch.from_numpy(frame.points_world).to(target)
        colors = torch.from_numpy(frame.point_colors).to(target)
        z = torch.from_numpy(np.maximum(frame.point_depths_m, 0.1).astype(np.float32)).to(target)
        focal = 0.5 * (frame.intrinsics.fx + frame.intrinsics.fy)
        base_scale = (z / focal).clamp_min(scale_clamp_min)
        log_scales = torch.log((base_scale[:, None] * scale_multiplier * torch.as_tensor(anisotropy, device=target)).clamp_min(scale_clamp_min))
        dc = ((colors - 0.5) / SH_C0).view(-1, 1, 3)
        opacity_logit = math.log(initial_opacity) - math.log1p(-initial_opacity)
        model = cls(
            points,
            dc,
            torch.full((len(points), 1), opacity_logit, dtype=torch.float32, device=target),
            log_scales,
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=target).repeat(len(points), 1),
            sh_rest=torch.zeros((len(points), (sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device=target),
            sh_degree=sh_degree,
            source_types=torch.from_numpy(frame.point_source_types).to(target),
            source_confidences=torch.from_numpy(frame.point_source_confidences).to(target),
        )
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
        rotations = torch.nn.functional.normalize(self.rotations, dim=1)
        return render(
            self.means3d,
            self.dc,
            self.sh_rest,
            self.opacities,
            self.scales,
            rotations,
            camera,
            sh_degree=self.sh_degree,
            debug=debug,
        )

    def append_frame(
        self,
        frame: BagFrame,
        *,
        optimizer: torch.optim.Optimizer,
        max_points: int | None = None,
        scale_clamp_min: float = 1e-4,
        scale_anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
        scale_multiplier: float = 2.0,
        growth_opacity: float = 0.1,
        alpha_gate: bool = True,
        pixel_dedup: bool = True,
    ) -> int:
        if not 0.0 < growth_opacity < 1.0:
            raise ValueError("growth_opacity must be within (0, 1)")
        if scale_multiplier <= 0 or not math.isfinite(scale_multiplier):
            raise ValueError("scale_multiplier must be positive and finite")
        candidate_indices = self._extension_candidate_indices(
            frame,
            alpha_gate=alpha_gate,
            pixel_dedup=pixel_dedup,
        )
        if max_points is not None and max_points < 1:
            raise ValueError("max_points must be positive")
        indices = candidate_indices if max_points is None else candidate_indices[:max_points]
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
            point_depths_m=frame.point_depths_m[indices],
            point_source_types=frame.point_source_types[indices],
            point_source_confidences=frame.point_source_confidences[indices],
        )
        additions = self._tensors_from_frame(
            candidate,
            scale_clamp_min=scale_clamp_min,
            scale_anisotropy=scale_anisotropy,
            scale_multiplier=scale_multiplier,
            opacity=growth_opacity,
        )
        self._append_tensors(additions, optimizer)
        return len(indices)

    def _tensors_from_frame(
        self,
        frame: BagFrame,
        *,
        scale_clamp_min: float,
        scale_anisotropy: tuple[float, float, float],
        scale_multiplier: float,
        opacity: float,
    ) -> tuple[torch.Tensor, ...]:
        target = self.means3d.device
        means = torch.from_numpy(frame.points_world).to(target)
        colors = torch.from_numpy(frame.point_colors).to(target)
        z = torch.from_numpy(np.maximum(frame.point_depths_m, 0.1).astype(np.float32)).to(target)
        base_scale = (z / (0.5 * (frame.intrinsics.fx + frame.intrinsics.fy))).clamp_min(scale_clamp_min)
        scales = base_scale[:, None] * scale_multiplier * torch.as_tensor(scale_anisotropy, dtype=torch.float32, device=target)
        opacity_logit = math.log(opacity) - math.log1p(-opacity)
        return (
            means,
            ((colors - 0.5) / SH_C0).view(-1, 1, 3),
            torch.zeros((len(means), (self.sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device=target),
            torch.full((len(means), 1), opacity_logit, device=target),
            torch.log(scales),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=target).repeat(len(means), 1),
            torch.from_numpy(frame.point_source_types).to(target),
            torch.from_numpy(frame.point_source_confidences).to(target),
        )

    def _extension_candidate_indices(
        self,
        frame: BagFrame,
        *,
        alpha_gate: bool,
        pixel_dedup: bool,
    ) -> np.ndarray:
        indices = np.arange(len(frame.points_world), dtype=np.int64)
        if not pixel_dedup:
            return indices
        camera_from_world = np.linalg.inv(frame.world_from_camera)
        points_h = np.concatenate(
            (frame.points_world.astype(np.float64), np.ones((len(frame.points_world), 1))),
            axis=1,
        )
        camera = (camera_from_world @ points_h.T).T[:, :3]
        z = camera[:, 2]
        safe_z = np.where(np.abs(z) > 1e-8, z, 1.0)
        pixels = np.column_stack(
            (
                np.floor(frame.intrinsics.fx * camera[:, 0] / safe_z + frame.intrinsics.cx),
                np.floor(frame.intrinsics.fy * camera[:, 1] / safe_z + frame.intrinsics.cy),
            )
        ).astype(np.int64)
        winners: dict[tuple[int, int], int] = {}
        for index, (u, v) in enumerate(pixels):
            key = (int(u), int(v))
            if key not in winners or z[index] < z[winners[key]]:
                winners[key] = index
        selected = np.asarray(list(winners.values()), dtype=np.int64)
        if not len(selected):
            return selected
        u, v = pixels[selected, 0], pixels[selected, 1]
        height, width = frame.intrinsics.height, frame.intrinsics.width
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        positive_depth = frame.point_depths_m[selected] > 0
        keep = in_image & positive_depth
        if alpha_gate and bool(keep.any()):
            if self.means3d.device.type != "cuda":
                raise RuntimeError("LIC alpha gating requires a CUDA Gaussian map")
            with torch.no_grad():
                rendered = self.render(frame.lic_camera())
            final_t = rendered.final_transmittance.detach().squeeze().cpu().numpy()
            if final_t.shape != (height, width):
                raise ValueError("LIC rasterizer final_transmittance has an unexpected shape")
            alpha = 1.0 - final_t
            clipped_u = np.clip(u, 0, width - 1)
            clipped_v = np.clip(v, 0, height - 1)
            keep &= alpha[clipped_v, clipped_u] < 0.99
        return selected[keep]

    def _append_tensors(self, additions: tuple[torch.Tensor, ...], optimizer: torch.optim.Optimizer) -> None:
        old_count = self.count
        old_parameters = {name: getattr(self, name) for name in self.PARAMETER_NAMES}
        for name, addition in zip(self.PARAMETER_NAMES, additions[: len(self.PARAMETER_NAMES)]):
            old = old_parameters[name]
            new = nn.Parameter(torch.cat((old.detach(), addition.detach()), dim=0))
            self._replace_optimizer_parameter(optimizer, old, new, old_count)
            setattr(self, name, new)
        self.source_types = torch.cat((self.source_types, additions[6].detach().to(self.source_types.device)), dim=0)
        self.source_confidences = torch.cat((self.source_confidences, additions[7].detach().to(self.source_confidences.device)), dim=0)
        self._validate()

    def prune_low_opacity(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        threshold: float,
    ) -> int:
        if threshold < 0 or not math.isfinite(threshold):
            raise ValueError("prune opacity threshold must be finite and non-negative")
        keep = self.opacities.detach().reshape(-1) >= threshold
        if int(keep.sum()) < 2:
            keep[:] = False
            keep[torch.topk(self.opacities.detach().reshape(-1), k=min(2, self.count)).indices] = True
        removed = self.count - int(keep.sum())
        if removed == 0:
            return 0
        old_count = self.count
        old_parameters = {name: getattr(self, name) for name in self.PARAMETER_NAMES}
        for name, old in old_parameters.items():
            new = nn.Parameter(old.detach()[keep])
            self._replace_optimizer_parameter(optimizer, old, new, old_count, keep=keep)
            setattr(self, name, new)
        self.source_types = self.source_types[keep]
        self.source_confidences = self.source_confidences[keep]
        self._validate()
        return removed

    @staticmethod
    def _replace_optimizer_parameter(
        optimizer: torch.optim.Optimizer,
        old: nn.Parameter,
        new: nn.Parameter,
        old_count: int,
        *,
        keep: torch.Tensor | None = None,
    ) -> None:
        for group in optimizer.param_groups:
            group["params"] = [new if parameter is old else parameter for parameter in group["params"]]
        state = optimizer.state.pop(old, {})
        migrated = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == old_count:
                migrated[key] = value[keep] if keep is not None else torch.cat(
                    (
                        value,
                        torch.zeros((new.shape[0] - old_count, *value.shape[1:]), dtype=value.dtype, device=value.device),
                    ),
                    dim=0,
                )
            else:
                migrated[key] = value
        optimizer.state[new] = migrated

    def _validate(self) -> None:
        count = self.means3d.shape[0]
        if count < 2:
            raise ValueError("GaussianMap requires at least two rows for the LIC reference kernel")
        if self.means3d.shape != (count, 3) or self.dc.shape != (count, 1, 3):
            raise ValueError("GaussianMap means3d/dc shapes are invalid")
        if self.sh_rest.shape != (count, (self.sh_degree + 1) ** 2 - 1, 3):
            raise ValueError("GaussianMap sh_rest shape is invalid")
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
