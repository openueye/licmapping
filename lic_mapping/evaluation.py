from __future__ import annotations

import json
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
from threading import Lock
from typing import Iterable
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import GaussianMap


EVALUATION_SCHEMA = "lic2-final-evaluation-v1"
SAGE_METRIC_SCHEMA = "sage-image-metrics-v1"
TORCHMETRICS_VERSION = "1.9.0"
TORCHVISION_VERSION_PREFIX = "0.20.1"
LPIPS_CALIBRATION_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
LPIPS_BACKBONE_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LPIPS_MODEL_ID = "alexnet-imagenet"
_LPIPS_BUILD_LOCK = Lock()
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*weights_only=False.*")


def default_lpips_backbone_path() -> Path:
    return Path(__file__).resolve().parents[2] / "SAGE-models" / "alexnet-owt-7be5be79.pth"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _force_torch_load_weights_only():
    original_torch_load = torch.load

    def load_with_weights_only(*args, **kwargs):
        kwargs.setdefault("weights_only", True)
        return original_torch_load(*args, **kwargs)

    torch.load = load_with_weights_only
    try:
        yield
    finally:
        torch.load = original_torch_load


class SAGEImageMetricEvaluator:
    """Compute SAGE's offline AlexNet LPIPS, PSNR, and Gaussian-window SSIM."""

    def __init__(self, device: torch.device, *, backbone: Path | None = None) -> None:
        self.device = torch.device(device)
        path = Path(backbone or default_lpips_backbone_path()).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SAGE LPIPS AlexNet weights do not exist: {path}")
        actual = _sha256_file(path)
        if actual != LPIPS_BACKBONE_SHA256:
            raise ValueError(
                "SAGE LPIPS AlexNet weights SHA-256 mismatch: "
                f"expected {LPIPS_BACKBONE_SHA256}, got {actual} at {path}"
            )
        calibration, torchmetrics_version, calibration_package = _lpips_calibration_path()
        try:
            import torchvision
        except ImportError as exc:
            raise RuntimeError("SAGE LPIPS evaluation requires torchvision") from exc
        torchvision_version = str(torchvision.__version__)
        if not torchvision_version.startswith(TORCHVISION_VERSION_PREFIX):
            raise ValueError(
                f"SAGE image metrics require torchvision {TORCHVISION_VERSION_PREFIX}.*, got {torchvision_version}"
            )
        self.identity = {
            "kind": "repository-offline",
            "model_id": LPIPS_MODEL_ID,
            "weights_sha256": actual,
            "evaluator_schema": SAGE_METRIC_SCHEMA,
            "torchmetrics_version": torchmetrics_version,
            "torchvision_version": torchvision_version,
            "calibration_sha256": LPIPS_CALIBRATION_SHA256,
            "calibration_package": calibration_package,
            "device": str(self.device),
            "dtype": "float32",
        }
        del calibration
        self.lpips = _build_lpips(_load_alexnet_features(path)).to(self.device)

    @torch.inference_mode()
    def __call__(self, rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        if rendered.shape != target.shape or rendered.ndim != 3 or rendered.shape[-1] != 3:
            raise ValueError("RGB tensors must share HxWx3 shape")
        rendered = rendered.to(self.device, dtype=torch.float32).clamp(0, 1)
        target = target.to(self.device, dtype=torch.float32).clamp(0, 1)
        mse = F.mse_loss(rendered, target)
        _, _, ssim = _sage_photometric_loss(rendered, target)
        lpips = self.lpips(
            rendered.permute(2, 0, 1).unsqueeze(0),
            target.permute(2, 0, 1).unsqueeze(0),
        )
        return {
            "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
            "ssim": float(ssim),
            "lpips": float(torch.as_tensor(lpips).mean()),
        }


def _lpips_calibration_path() -> tuple[Path, str, str]:
    torchmetrics = importlib.import_module("torchmetrics")
    version = str(torchmetrics.__version__)
    if version != TORCHMETRICS_VERSION:
        raise ValueError(f"SAGE image metrics require torchmetrics=={TORCHMETRICS_VERSION}, got {version}")
    package_path = Path(torchmetrics.__file__).resolve().parent
    calibration = package_path / "functional" / "image" / "lpips_models" / "alex.pth"
    if not calibration.is_file() or _sha256_file(calibration) != LPIPS_CALIBRATION_SHA256:
        raise ValueError(f"TorchMetrics LPIPS Alex calibration is missing or mismatched: {calibration}")
    return calibration, version, f"torchmetrics=={version}:functional/image/lpips_models/alex.pth"


def _load_alexnet_features(path: Path) -> torch.nn.Sequential:
    try:
        import torchvision

        model = torchvision.models.alexnet(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("AlexNet checkpoint must contain a state dictionary")
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"SAGE AlexNet backbone is structurally incompatible: {path}") from exc
    return model.features


def _build_lpips(features: torch.nn.Sequential) -> torch.nn.Module:
    lpips_module = importlib.import_module("torchmetrics.functional.image.lpips")
    metric_type = getattr(importlib.import_module("torchmetrics.image.lpip"), "LearnedPerceptualImagePatchSimilarity")
    original_resolver = lpips_module._get_tv_model_features

    def explicit_features(net: str, pretrained: bool = False) -> torch.nn.Sequential:
        if net != "alexnet" or not pretrained:
            raise ValueError(f"SAGE LPIPS only supports the local pretrained AlexNet, got {net}")
        return features

    with _LPIPS_BUILD_LOCK:
        lpips_module._get_tv_model_features = explicit_features
        try:
            with _force_torch_load_weights_only():
                return metric_type(net_type="alex", normalize=True).eval().requires_grad_(False)
        finally:
            lpips_module._get_tv_model_features = original_resolver


def _sage_photometric_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    ssim_weight: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = rendered.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    coordinates = torch.arange(11, dtype=x.dtype, device=x.device) - 5
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).view(1, 1, 11, 11).expand(3, 1, 11, 11).contiguous()
    mu_x = F.conv2d(x, window, padding=5, groups=3)
    mu_y = F.conv2d(y, window, padding=5, groups=3)
    sigma_x = F.conv2d(x * x, window, padding=5, groups=3) - mu_x.square()
    sigma_y = F.conv2d(y * y, window, padding=5, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(x * y, window, padding=5, groups=3) - mu_x * mu_y
    ssim = (((2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)) /
            ((mu_x.square() + mu_y.square() + 0.01**2) * (sigma_x + sigma_y + 0.03**2))).mean()
    l1 = F.l1_loss(rendered, target)
    image = (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim.clamp(-1, 1))
    return image, l1, ssim


def evaluate_final_map(
    model: GaussianMap,
    keyframes: Iterable[object],
    output_dir: Path,
    *,
    lpips_backbone: Path | None = None,
) -> dict[str, object]:
    """Write LIC2-compatible final metrics and inspection artifacts.

    The evaluator consumes retained keyframe views, so evaluation remains
    bounded by the number of training views rather than the complete ROSBAG.
    Raw arrays are saved alongside PNGs to keep visualizations auditable.
    """

    root = Path(output_dir)
    for name in ("renders/rgb", "renders/target", "renders/depth", "renders/alpha", "renders/error", "arrays"):
        (root / name).mkdir(parents=True, exist_ok=True)
    image_metrics = SAGEImageMetricEvaluator(model.means3d.device, backbone=lpips_backbone)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for view in keyframes:
            output = model.render(view.camera)
            rendered_rgb = output.rgb.clamp(0, 1)
            target_rgb = torch.from_numpy(view.rgb).permute(2, 0, 1).to(model.means3d.device)
            rendered_depth = output.depth.squeeze()
            target_depth = torch.from_numpy(view.depth_m).to(model.means3d.device)
            alpha = (1.0 - output.final_transmittance.squeeze()).clamp(0, 1)
            quality = image_metrics(
                rendered_rgb.permute(1, 2, 0),
                target_rgb.permute(1, 2, 0),
            )
            depth_valid = (target_depth > 0) & (rendered_depth > 0) & torch.isfinite(rendered_depth)
            if bool(depth_valid.any()):
                depth_mae = float((rendered_depth[depth_valid] - target_depth[depth_valid]).abs().mean())
            else:
                depth_mae = None
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
                "psnr": quality["psnr"],
                "ssim": quality["ssim"],
                "lpips": quality["lpips"],
                "depth_mae_m": depth_mae,
                "depth_valid_pixels": int(depth_valid.sum()),
                "depth_target_pixels": int((target_depth > 0).sum()),
                "alpha_mean": float(alpha.mean()),
                "alpha_supported_pixels": int((alpha > 0.01).sum()),
            })
    _write_gaussian_artifacts(model, root / "map")
    metrics = {
        "schema_version": EVALUATION_SCHEMA,
        "evaluation_protocol": SAGE_METRIC_SCHEMA,
        "frame_selection": "retained_keyframes",
        "gaussian_count": model.count,
        "keyframes": rows,
        "aggregate": _aggregate(rows),
        "metric_identity": image_metrics.identity,
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


__all__ = ["EVALUATION_SCHEMA", "SAGEImageMetricEvaluator", "default_lpips_backbone_path", "evaluate_final_map"]
