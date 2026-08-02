from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lic_mapping.rasterizer import LicCamera, render


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


def _direct_native(
    means3d: torch.Tensor,
    dc: torch.Tensor,
    sh: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    camera: LicCamera,
    background: torch.Tensor,
    scale_modifier: float,
):
    binding = importlib.import_module("lic_mapping._C")
    viewmatrix, projection, camera_center, tanfovx, tanfovy, limx_neg, limx_pos, limy_neg, limy_pos = _reference_camera_tensors(
        camera,
        device=means3d.device,
    )
    return binding.rasterize(
        means3d.contiguous(),
        torch.zeros_like(means3d, requires_grad=True),
        dc.contiguous(),
        sh.contiguous(),
        opacities.contiguous(),
        scales.contiguous(),
        F.normalize(rotations, dim=1).contiguous(),
        background.contiguous(),
        viewmatrix,
        (viewmatrix @ projection).contiguous(),
        camera_center,
        camera.height,
        camera.width,
        tanfovx,
        tanfovy,
        limx_neg,
        limx_pos,
        limy_neg,
        limy_pos,
        2,
        scale_modifier,
        0.0,
        False,
        False,
        False,
    )


def _reference_camera_tensors(camera: LicCamera, *, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Independently reproduce Gaussian-LIC ``Camera`` in camera.h."""
    pose = camera.world_from_camera.detach().cpu().numpy().astype(np.float64)
    world_to_camera = np.linalg.inv(pose)
    # Camera::setWorldViewTransform stores the column-major view matrix.
    viewmatrix = torch.as_tensor(world_to_camera.T.copy(), dtype=torch.float32, device=device)

    # Camera::setProjectionMatrix fills P then stores P.transpose(0, 1).
    p = np.zeros((4, 4), dtype=np.float32)
    fovx = 2.0 * np.arctan(camera.width / (2.0 * camera.fx))
    fovy = 2.0 * np.arctan(camera.height / (2.0 * camera.fy))
    p[0, 0] = 1.0 / np.tan(fovx / 2.0)
    p[1, 1] = 1.0 / np.tan(fovy / 2.0)
    p[0, 2] = (2.0 * camera.cx - camera.width) / camera.width
    p[1, 2] = (2.0 * camera.cy - camera.height) / camera.height
    p[2, 2] = camera.zfar / (camera.zfar - camera.znear)
    p[2, 3] = -(camera.zfar * camera.znear) / (camera.zfar - camera.znear)
    p[3, 2] = 1.0
    projection = torch.as_tensor(p.T.copy(), dtype=torch.float32, device=device)
    return (
        viewmatrix,
        projection,
        torch.as_tensor(pose[:3, 3].copy(), dtype=torch.float32, device=device),
        camera.width / (2.0 * camera.fx),
        camera.height / (2.0 * camera.fy),
        -0.15 * camera.width / camera.fx - camera.cx / camera.fx,
        1.15 * camera.width / camera.fx - camera.cx / camera.fx,
        -0.15 * camera.height / camera.fy - camera.cy / camera.fy,
        1.15 * camera.height / camera.fy - camera.cy / camera.fy,
    )


def test_camera_tensors_match_independent_gaussian_lic_camera_convention() -> None:
    pose = torch.tensor(
        [[0.0, -1.0, 0.0, 0.4], [1.0, 0.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera = LicCamera(32, 24, 25.0, 24.0, 15.5, 11.5, pose)
    actual_view, actual_projection, actual_center, *actual_limits = camera.tensors(device="cpu")
    expected_view, expected_projection, expected_center, *expected_limits = _reference_camera_tensors(
        camera,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(actual_view, expected_view, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_projection, expected_projection, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_view @ actual_projection, expected_view @ expected_projection, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_center, expected_center, rtol=0, atol=0)
    assert actual_limits == pytest.approx(expected_limits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_adapter_matches_direct_reference_binding_with_nonidentity_pose() -> None:
    device = torch.device("cuda")
    pose = torch.eye(4)
    pose[0, 3] = 0.15
    pose[1, 3] = -0.05
    camera = LicCamera(32, 24, 25.0, 24.0, 15.5, 11.5, pose)
    background = torch.tensor([0.05, 0.1, 0.15], device=device)
    adapter_inputs = _inputs(device)
    direct_inputs = tuple(value.detach().clone().requires_grad_() for value in adapter_inputs)

    adapter = render(*adapter_inputs, camera, sh_degree=2, background=background, scale_modifier=0.8)
    direct_rgb, direct_radii, direct_depth, direct_transmittance = _direct_native(
        *direct_inputs, camera, background, 0.8
    )

    torch.testing.assert_close(adapter.rgb, direct_rgb, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(adapter.depth, direct_depth, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(adapter.final_transmittance, direct_transmittance, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(adapter.radii, direct_radii, rtol=0, atol=0)

    adapter_loss = adapter.rgb.square().mean() + adapter.depth.square().mean() + adapter.final_transmittance.square().mean()
    direct_loss = direct_rgb.square().mean() + direct_depth.square().mean() + direct_transmittance.square().mean()
    adapter_loss.backward()
    direct_loss.backward()
    for adapter_input, direct_input in zip(adapter_inputs, direct_inputs):
        assert adapter_input.grad is not None
        assert direct_input.grad is not None
        torch.testing.assert_close(adapter_input.grad, direct_input.grad, rtol=5e-5, atol=5e-6)
