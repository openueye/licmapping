from __future__ import annotations

import pytest
import torch


lic_mapping = pytest.importorskip("lic_mapping")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_small_scene_forward_backward() -> None:
    device = torch.device("cuda")
    camera = lic_mapping.LicCamera(
        width=32,
        height=32,
        fx=24.0,
        fy=24.0,
        cx=15.5,
        cy=15.5,
        world_from_camera=torch.eye(4),
    )
    means3d = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 2.5]],
        device=device,
        requires_grad=True,
    )
    color = torch.tensor([[1.0, 0.2, 0.1], [0.2, 0.4, 0.8]], device=device)
    c0 = 0.28209479177387814
    dc = ((color - 0.5) / c0).view(2, 1, 3).requires_grad_()
    sh = torch.zeros((2, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    # LIC's rasterizer receives activated opacity; GaussianModel::getOpacity()
    # applies sigmoid before calling the C++ wrapper.
    opacities = torch.tensor([[0.5], [0.0]], device=device, requires_grad=True)
    scales = torch.full((2, 3), 0.08, device=device, requires_grad=True)
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        device=device,
        requires_grad=True,
    )

    output = lic_mapping.render(
        means3d,
        dc,
        sh,
        opacities,
        scales,
        rotations,
        camera,
        sh_degree=3,
    )

    assert output.rgb.shape == (3, 32, 32)
    assert output.depth.shape == (32, 32)
    assert output.radii.shape == (2,)
    assert bool(output.visible.any())
    assert bool(torch.isfinite(output.rgb).all())
    assert float(output.rgb.max()) > 0.01
    assert bool(torch.isfinite(output.depth).all())
    assert bool((output.depth > 0).any())
    assert bool((1.0 - output.final_transmittance > 0.01).any())

    loss = output.rgb.square().mean() + output.depth.square().mean()
    loss.backward()
    for parameter in (means3d, dc, sh, opacities, scales, rotations):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
