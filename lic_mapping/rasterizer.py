from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import _C


@dataclass(frozen=True)
class LicCamera:
    """Camera tensors using the same column-major convention as LIC."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_from_camera: torch.Tensor
    znear: float = 0.01
    zfar: float = 100.0

    def tensors(self, *, device: torch.device | str) -> tuple[torch.Tensor, ...]:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if not (0.0 < self.znear < self.zfar):
            raise ValueError("camera depth range must satisfy 0 < znear < zfar")
        if not all(
            math.isfinite(value)
            for value in (self.fx, self.fy, self.cx, self.cy, self.znear, self.zfar)
        ):
            raise ValueError("camera intrinsics and depth range must be finite")
        target = torch.device(device)
        pose = self.world_from_camera.to(device=target, dtype=torch.float32)
        if pose.shape != (4, 4):
            raise ValueError("world_from_camera must have shape [4, 4]")
        if not torch.isfinite(pose).all():
            raise ValueError("world_from_camera must be finite")

        world_to_camera = torch.linalg.inv(pose)
        viewmatrix = world_to_camera.transpose(0, 1).contiguous()
        projection = torch.zeros((4, 4), dtype=torch.float32, device=target)
        projection[0, 0] = 2.0 * self.fx / self.width
        projection[1, 1] = 2.0 * self.fy / self.height
        projection[0, 2] = (2.0 * self.cx - self.width) / self.width
        projection[1, 2] = (2.0 * self.cy - self.height) / self.height
        projection[2, 2] = self.zfar / (self.zfar - self.znear)
        projection[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        projection[3, 2] = 1.0
        projmatrix = projection.transpose(0, 1).contiguous()

        camera_center = pose[:3, 3].contiguous()
        tanfovx = self.width / (2.0 * self.fx)
        tanfovy = self.height / (2.0 * self.fy)
        limx_neg = -0.15 * self.width / self.fx - self.cx / self.fx
        limx_pos = 1.15 * self.width / self.fx - self.cx / self.fx
        limy_neg = -0.15 * self.height / self.fy - self.cy / self.fy
        limy_pos = 1.15 * self.height / self.fy - self.cy / self.fy
        return (
            viewmatrix,
            projmatrix,
            camera_center,
            tanfovx,
            tanfovy,
            limx_neg,
            limx_pos,
            limy_neg,
            limy_pos,
        )


@dataclass(frozen=True)
class LicRenderOutput:
    rgb: torch.Tensor
    radii: torch.Tensor
    depth: torch.Tensor
    final_transmittance: torch.Tensor

    @property
    def visible(self) -> torch.Tensor:
        return self.radii > 0


def render(
    means3d: torch.Tensor,
    dc: torch.Tensor,
    sh: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    camera: LicCamera,
    *,
    sh_degree: int = 3,
    background: torch.Tensor | None = None,
    scale_modifier: float = 1.0,
    lambda_erank: float = 0.0,
    prefiltered: bool = False,
    debug: bool = False,
    no_color: bool = False,
) -> LicRenderOutput:
    """Render LIC Gaussian tensors while preserving autograd through CUDA.

    ``opacities`` must already be in ``[0, 1]``. This matches LIC's C++
    ``GaussianModel::getOpacity()`` output; callers that store logits should
    pass ``torch.sigmoid(opacity_logits)``.
    """
    if means3d.device.type != "cuda":
        raise ValueError("LIC rasterizer requires CUDA tensors")
    if means3d.dtype != torch.float32:
        raise ValueError("LIC rasterizer currently requires float32 tensors")
    if background is None:
        background = torch.zeros(3, dtype=torch.float32, device=means3d.device)
    else:
        background = background.to(device=means3d.device, dtype=torch.float32).contiguous()
    means3d = means3d.contiguous()
    dc = dc.contiguous()
    sh = sh.contiguous()
    opacities = opacities.contiguous()
    scales = scales.contiguous()
    rotations = rotations.contiguous()
    means2d = torch.zeros_like(means3d, requires_grad=True)
    tensors = camera.tensors(device=means3d.device)
    output = _C.rasterize(
        means3d,
        means2d,
        dc,
        sh,
        opacities,
        scales,
        rotations,
        background,
        tensors[0],
        tensors[1],
        tensors[2],
        camera.height,
        camera.width,
        tensors[3],
        tensors[4],
        tensors[5],
        tensors[6],
        tensors[7],
        tensors[8],
        sh_degree,
        scale_modifier,
        lambda_erank,
        prefiltered,
        debug,
        no_color,
    )
    return LicRenderOutput(
        rgb=output[0],
        radii=output[1],
        depth=output[2],
        final_transmittance=output[3],
    )
