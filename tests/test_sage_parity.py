from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lic_mapping.rasterizer import LicCamera, _load_sage_backend, _sage_settings, render


def _inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    means3d = torch.tensor(
        [[-0.45, 0.10, 2.0], [0.35, -0.15, 2.4], [-0.10, 0.35, 2.8], [0.20, 0.20, 3.2]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    colors = torch.tensor(
        [[0.9, 0.2, 0.1], [0.1, 0.8, 0.2], [0.2, 0.3, 0.9], [0.7, 0.6, 0.2]],
        dtype=torch.float32,
        device=device,
    )
    sh_degree = 2
    c0 = 0.28209479177387814
    dc = ((colors - 0.5) / c0).view(4, 1, 3).requires_grad_()
    sh = torch.linspace(-0.03, 0.03, 4 * ((sh_degree + 1) ** 2 - 1) * 3, device=device)
    sh = sh.reshape(4, (sh_degree + 1) ** 2 - 1, 3).requires_grad_()
    opacities = torch.tensor([[0.35], [0.5], [0.65], [0.8]], device=device, requires_grad=True)
    scales = torch.tensor(
        [[0.08, 0.07, 0.09], [0.10, 0.09, 0.08], [0.07, 0.11, 0.09], [0.09, 0.08, 0.12]],
        device=device,
        requires_grad=True,
    )
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.98, 0.1, 0.0, 0.0], [0.97, 0.0, 0.2, 0.0], [0.96, 0.0, 0.0, 0.28]],
        device=device,
        requires_grad=True,
    )
    return means3d, dc, sh, opacities, scales, rotations


def _direct_sage(
    means3d: torch.Tensor,
    dc: torch.Tensor,
    sh: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    camera: LicCamera,
    background: torch.Tensor,
    scale_modifier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    backend = _load_sage_backend()
    settings = _sage_settings(
        backend,
        camera,
        device=means3d.device,
        sh_degree=2,
        background=background,
    )._replace(scale_modifier=scale_modifier, prefiltered=False)
    rasterizer = backend.GaussianRasterizer(settings)
    means3d = means3d.contiguous()
    shs = torch.cat((dc.contiguous(), sh.contiguous()), dim=1)
    opacities = opacities.contiguous()
    scales = scales.contiguous()
    rotations = F.normalize(rotations, dim=1).contiguous()
    rgb, radii, _ = rasterizer(
        means3D=means3d,
        means2D=torch.zeros_like(means3d, requires_grad=True),
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
    return rgb, silhouette_render[0], alpha, radii


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sage_adapter_matches_direct_extension_outputs_and_gradients() -> None:
    device = torch.device("cuda")
    camera = LicCamera(
        width=32,
        height=24,
        fx=25.0,
        fy=24.0,
        cx=15.5,
        cy=11.5,
        world_from_camera=torch.eye(4),
    )
    background = torch.tensor([0.05, 0.1, 0.15], device=device)
    adapter_inputs = _inputs(device)
    direct_inputs = tuple(value.detach().clone().requires_grad_() for value in adapter_inputs)

    adapter = render(
        *adapter_inputs,
        camera,
        sh_degree=2,
        background=background,
        scale_modifier=0.8,
    )
    direct_rgb, direct_depth, direct_alpha, direct_radii = _direct_sage(
        *direct_inputs,
        camera,
        background,
        0.8,
    )

    torch.testing.assert_close(adapter.rgb, direct_rgb, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(adapter.depth, direct_depth, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(1.0 - adapter.final_transmittance, direct_alpha, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(adapter.radii, direct_radii, rtol=0, atol=0)

    adapter_loss = adapter.rgb.square().mean() + adapter.depth.square().mean() + (1.0 - adapter.final_transmittance).square().mean()
    direct_loss = direct_rgb.square().mean() + direct_depth.square().mean() + direct_alpha.square().mean()
    adapter_loss.backward()
    direct_loss.backward()
    for adapter_input, direct_input in zip(adapter_inputs, direct_inputs):
        assert adapter_input.grad is not None
        assert direct_input.grad is not None
        torch.testing.assert_close(adapter_input.grad, direct_input.grad, rtol=5e-5, atol=5e-6)
