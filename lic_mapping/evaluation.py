from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import GaussianMap


EVALUATION_SCHEMA = "lic2-final-evaluation-v1"


def evaluate_final_map(
    model: GaussianMap,
    keyframes: Iterable[object],
    output_dir: Path,
    *,
    lpips_model: Path | None = None,
) -> dict[str, object]:
    """Write LIC2-compatible final metrics and inspection artifacts.

    The evaluator consumes retained keyframe views, so evaluation remains
    bounded by the number of training views rather than the complete ROSBAG.
    Raw arrays are saved alongside PNGs to keep visualizations auditable.
    """

    root = Path(output_dir)
    for name in ("renders/rgb", "renders/target", "renders/depth", "renders/alpha", "renders/error", "arrays"):
        (root / name).mkdir(parents=True, exist_ok=True)
    lpips = _load_lpips(lpips_model, model.means3d.device) if lpips_model is not None else None
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for view in keyframes:
            output = model.render(view.camera)
            rendered_rgb = model.correct_exposure(output.rgb).clamp(0, 1)
            target_rgb = torch.from_numpy(view.rgb).permute(2, 0, 1).to(model.means3d.device)
            rendered_depth = output.depth.squeeze()
            target_depth = torch.from_numpy(view.depth_m).to(model.means3d.device)
            alpha = (1.0 - output.final_transmittance.squeeze()).clamp(0, 1)
            rgb_mse = F.mse_loss(rendered_rgb, target_rgb)
            depth_valid = (target_depth > 0) & (rendered_depth > 0) & torch.isfinite(rendered_depth)
            if bool(depth_valid.any()):
                depth_mae = float((rendered_depth[depth_valid] - target_depth[depth_valid]).abs().mean())
            else:
                depth_mae = None
            frame_lpips = _compute_lpips(lpips, rendered_rgb, target_rgb) if lpips is not None else None
            stem = f"{int(view.index):06d}"
            rendered_np = rendered_rgb.permute(1, 2, 0).cpu().numpy()
            target_np = target_rgb.permute(1, 2, 0).cpu().numpy()
            _write_rgb(root / "renders/rgb" / f"{stem}.png", rendered_np)
            _write_rgb(root / "renders/target" / f"{stem}.png", target_np)
            rendered_depth_np = rendered_depth.cpu().numpy().astype(np.float32)
            target_depth_np = target_depth.cpu().numpy().astype(np.float32)
            alpha_np = alpha.cpu().numpy().astype(np.float32)
            np.save(root / "arrays" / f"{stem}_rendered_depth.npy", rendered_depth_np)
            np.save(root / "arrays" / f"{stem}_target_depth.npy", target_depth_np)
            np.save(root / "arrays" / f"{stem}_alpha.npy", alpha_np)
            _write_depth(root / "renders/depth" / f"{stem}.png", rendered_depth_np)
            _write_depth(root / "renders/depth" / f"{stem}_target.png", target_depth_np)
            cv2.imwrite(str(root / "renders/alpha" / f"{stem}.png"), (alpha_np * 255).clip(0, 255).astype(np.uint8))
            error = np.abs(rendered_np - target_np).mean(axis=2)
            cv2.imwrite(str(root / "renders/error" / f"{stem}.png"), (error * 255).clip(0, 255).astype(np.uint8))
            rows.append({
                "frame_index": int(view.index),
                "psnr": _psnr(float(rgb_mse)),
                "ssim": float(_ssim(rendered_rgb, target_rgb)),
                "lpips": frame_lpips,
                "depth_mae_m": depth_mae,
                "depth_valid_pixels": int(depth_valid.sum()),
                "depth_target_pixels": int((target_depth > 0).sum()),
                "alpha_mean": float(alpha.mean()),
                "alpha_supported_pixels": int((alpha > 0.01).sum()),
            })
    _write_gaussian_artifacts(model, root / "map")
    metrics = {
        "schema_version": EVALUATION_SCHEMA,
        "gaussian_count": model.count,
        "exposure": model.exposure.detach().cpu().tolist(),
        "keyframes": rows,
        "aggregate": _aggregate(rows),
        "lpips": {
            "requested": lpips_model is not None,
            "available": lpips is not None,
            "model": str(lpips_model) if lpips_model is not None else None,
        },
    }
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"keyframes": 0}
    result: dict[str, object] = {"keyframes": len(rows)}
    for name in ("psnr", "ssim", "lpips", "depth_mae_m", "alpha_mean"):
        values = [float(row[name]) for row in rows if row[name] is not None and np.isfinite(float(row[name]))]
        result[name] = float(np.mean(values)) if values else None
    result["depth_valid_pixels"] = int(sum(int(row["depth_valid_pixels"]) for row in rows))
    result["depth_target_pixels"] = int(sum(int(row["depth_target_pixels"]) for row in rows))
    return result


def _psnr(mse: float) -> float:
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def _ssim(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rendered = rendered.unsqueeze(0)
    target = target.unsqueeze(0)
    mu_rendered = F.avg_pool2d(rendered, 3, stride=1, padding=1)
    mu_target = F.avg_pool2d(target, 3, stride=1, padding=1)
    sigma_rendered = F.avg_pool2d(rendered * rendered, 3, stride=1, padding=1) - mu_rendered.square()
    sigma_target = F.avg_pool2d(target * target, 3, stride=1, padding=1) - mu_target.square()
    sigma_cross = F.avg_pool2d(rendered * target, 3, stride=1, padding=1) - mu_rendered * mu_target
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_rendered * mu_target + c1) * (2 * sigma_cross + c2)) / (
        (mu_rendered.square() + mu_target.square() + c1)
        * (sigma_rendered + sigma_target + c2)
    )
    return score.clamp(0, 1).mean()


def _load_lpips(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"LPIPS TorchScript model does not exist: {source}")
    try:
        return torch.jit.load(str(source), map_location=device).eval()
    except Exception as exc:
        raise RuntimeError(f"Cannot load LIC2 LPIPS model: {source}") from exc


def _compute_lpips(model: torch.jit.ScriptModule, rendered: torch.Tensor, target: torch.Tensor) -> float:
    try:
        value = model(rendered.unsqueeze(0), target.unsqueeze(0))
    except Exception:
        value = model([rendered.unsqueeze(0), target.unsqueeze(0)])
    if isinstance(value, (tuple, list)):
        value = value[0]
    return float(torch.as_tensor(value).mean())


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    image = (np.asarray(rgb).clip(0, 1) * 255).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _write_depth(path: Path, depth: np.ndarray) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        low, high = np.percentile(depth[valid], (1, 99))
        normalized = ((depth - low) / max(float(high - low), 1e-6)).clip(0, 1)
        image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        image[~valid] = 0
    cv2.imwrite(str(path), image)


def _write_gaussian_artifacts(model: GaussianMap, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "means3d": model.means3d.detach().cpu().numpy(),
        "dc": model.dc.detach().cpu().numpy(),
        "sh_rest": model.sh_rest.detach().cpu().numpy(),
        "opacity": model.opacities.detach().cpu().numpy(),
        "scales": model.scales.detach().cpu().numpy(),
        "rotations": torch.nn.functional.normalize(model.rotations, dim=1).detach().cpu().numpy(),
        "source_types": model.source_types.detach().cpu().numpy(),
        "source_confidences": model.source_confidences.detach().cpu().numpy(),
    }
    np.savez_compressed(output_dir / "gaussians.npz", **arrays)
    colors = np.clip(arrays["dc"][:, 0, :] * 0.28209479177387814 + 0.5, 0, 1)
    _write_ply(output_dir / "point_cloud.ply", arrays["means3d"], colors, arrays["opacity"], arrays["scales"])


def _write_ply(path: Path, means: np.ndarray, colors: np.ndarray, opacity: np.ndarray, scales: np.ndarray) -> None:
    count = len(means)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float opacity\nproperty float scale_x\nproperty float scale_y\nproperty float scale_z\nend_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("opacity", "<f4"), ("scale_x", "<f4"), ("scale_y", "<f4"), ("scale_z", "<f4"),
    ])
    vertices = np.empty(count, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = (colors * 255).round().astype(np.uint8).T
    vertices["opacity"] = np.asarray(opacity).reshape(-1)
    vertices["scale_x"], vertices["scale_y"], vertices["scale_z"] = scales[:, 0], scales[:, 1], scales[:, 2]
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


__all__ = ["EVALUATION_SCHEMA", "evaluate_final_map"]
