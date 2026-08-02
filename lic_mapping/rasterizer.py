from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys

import torch


RENDERER_ID = "sage.diff_gaussian_rasterization"
RENDERER_ALIGNMENT = "approximate_substitution"
RENDERER_REFERENCE = "Gaussian-LIC/src/rasterizer"


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


def render_lic(
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
    """Render with the original LIC Gaussian tensors and CUDA extension.

    ``opacities`` must already be in ``[0, 1]``. This matches LIC's C++
    ``GaussianModel::getOpacity()`` output; callers that store logits should
    pass ``torch.sigmoid(opacity_logits)``.
    """
    from . import _C

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


def _load_sage_backend():
    try:
        backend = import_module("diff_gaussian_rasterization")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SAGE CUDA rasterizer is unavailable in the active environment. "
            "Build it with: python -m pip install --no-build-isolation --no-deps "
            "/home/DL/Projects/02_Thesis/00_Baselines/SAGE/"
            "third_party/diff-gaussian-rasterization-w-depth"
        ) from exc
    environment_root = Path(sys.prefix).resolve()
    package_path = Path(backend.__file__).resolve()
    extension_path = Path(backend._C.__file__).resolve()
    if not package_path.is_relative_to(environment_root) or not extension_path.is_relative_to(environment_root):
        raise RuntimeError(
            "SAGE CUDA rasterizer must load from the active Conda environment; "
            f"got {extension_path}"
        )
    return backend


def _sage_settings(
    backend: object,
    camera: LicCamera,
    *,
    device: torch.device,
    sh_degree: int,
    background: torch.Tensor | None,
) -> object:
    viewmatrix, projection, camera_center, tanfovx, tanfovy, *_ = camera.tensors(device=device)
    # SAGE's backend expects the view-projection matrix, while LicCamera's
    # second tensor is the camera projection matrix kept for the LIC binding.
    projmatrix = (viewmatrix @ projection).contiguous()
    values = {
        "image_height": camera.height,
        "image_width": camera.width,
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": (
            torch.zeros(3, dtype=torch.float32, device=device)
            if background is None
            else background.to(device=device, dtype=torch.float32).contiguous()
        ),
        "scale_modifier": 1.0,
        "viewmatrix": viewmatrix,
        "projmatrix": projmatrix,
        "sh_degree": sh_degree,
        "campos": camera_center,
        "prefiltered": False,
    }
    fields = getattr(backend.GaussianRasterizationSettings, "_fields", ())
    for optional in ("debug", "antialiasing"):
        if optional in fields:
            values[optional] = False
    return backend.GaussianRasterizationSettings(**values)


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
    """Render the LIC map through SAGE's approximate CUDA substitution.

    The mapping state remains LIC-native: ``dc`` and ``sh`` are concatenated
    and sent through SAGE's SH path, so gradients still reach both color
    parameter groups. SAGE's silhouette pass supplies the depth and alpha
    contract consumed by LIC2's loss, alpha gate, and final evaluator.
    """
    del lambda_erank, debug, no_color
    if means3d.device.type != "cuda":
        raise ValueError("SAGE rasterizer requires CUDA tensors")
    if means3d.dtype != torch.float32:
        raise ValueError("SAGE rasterizer currently requires float32 tensors")
    if sh_degree < 0 or sh_degree > 3:
        raise ValueError("sh_degree must be within [0, 3]")
    if dc.ndim != 3 or dc.shape[1:] != (1, 3):
        raise ValueError("dc must have shape [N, 1, 3]")
    expected_sh = (sh_degree + 1) ** 2 - 1
    if sh_degree == 0 and sh.numel() == 0:
        sh = torch.empty((means3d.shape[0], 0, 3), dtype=dc.dtype, device=dc.device)
    if sh.shape != (means3d.shape[0], expected_sh, 3):
        raise ValueError("sh has an unexpected shape for sh_degree")
    if means3d.shape[0] < 2:
        raise ValueError("SAGE rasterizer requires at least two Gaussian rows")

    backend = _load_sage_backend()
    settings = _sage_settings(
        backend,
        camera,
        device=means3d.device,
        sh_degree=sh_degree,
        background=background,
    )
    settings = settings._replace(scale_modifier=scale_modifier, prefiltered=prefiltered)
    rasterizer = backend.GaussianRasterizer(settings)
    means3d = means3d.contiguous()
    means2d = torch.zeros_like(means3d, requires_grad=True)
    shs = torch.cat((dc.contiguous(), sh.contiguous()), dim=1)
    rotations = torch.nn.functional.normalize(rotations, dim=1).contiguous()
    opacities = opacities.contiguous()
    scales = scales.contiguous()

    rgb, radii, _ = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=shs,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )

    pose = camera.world_from_camera.to(device=means3d.device, dtype=torch.float32)
    world_to_camera = torch.linalg.inv(pose)
    points_h = torch.cat(
        (means3d, torch.ones((means3d.shape[0], 1), dtype=means3d.dtype, device=means3d.device)),
        dim=1,
    )
    camera_z = (points_h @ world_to_camera.T)[:, 2]
    silhouette = torch.stack((camera_z, torch.ones_like(camera_z), camera_z.square()), dim=1)
    silhouette_render, _, _ = backend.GaussianRasterizer(settings)(
        means3D=means3d,
        means2D=torch.zeros_like(means3d),
        colors_precomp=silhouette.contiguous(),
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )
    alpha = silhouette_render[1].clamp(0, 1)
    return LicRenderOutput(
        rgb=rgb,
        radii=radii,
        depth=silhouette_render[0],
        final_transmittance=1.0 - alpha,
    )
