from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = 1
_SECTION_KEYS = {
    "input": {
        "rosbag", "calibration", "device", "frame_limit", "resize_width", "resize_height",
        "depth_adapter", "sync_tolerance_ms", "cloud_tolerance_ms",
    },
    "output": {"checkpoint", "artifact_dir", "save_artifacts"},
    "training": {
        "iterations_per_frame", "keyframe_every", "replay_keyframes", "max_initial_points",
        "max_new_points_per_frame", "initial_opacity", "growth_opacity", "scale_clamp_min",
        "scale_multiplier", "scale_anisotropy", "sh_degree",
        "prune_opacity_threshold", "prune_every_n_keyframes", "learning_rate_means",
        "learning_rate_dc", "learning_rate_opacity", "learning_rate_scales",
        "learning_rate_rotations", "rgb_weight", "lambda_dssim", "optimize_depth",
        "depth_weight", "iteration_decay", "depth_completion_patch_size",
        "depth_completion_max_depth_m", "depth_completion_confidence",
    },
    "spnet": {"engine", "torchscript", "weights", "source", "alignment"},
    "evaluation": {"lpips_backbone"},
}


def load_yaml_config(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate one LIC mapping YAML configuration."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configurations require PyYAML; install it in the active environment") from exc
    config_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read LIC mapping config: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("LIC mapping config root must be a YAML mapping")
    if payload.get("schema_version", CONFIG_SCHEMA_VERSION) != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported LIC mapping config schema: {payload.get('schema_version')}")
    unknown_sections = set(payload) - {"schema_version", *_SECTION_KEYS}
    if unknown_sections:
        raise ValueError(f"Unknown LIC mapping config sections: {', '.join(sorted(unknown_sections))}")
    result: dict[str, dict[str, Any]] = {}
    for section, allowed_keys in _SECTION_KEYS.items():
        values = payload.get(section, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"LIC mapping config section '{section}' must be a YAML mapping")
        unknown_keys = set(values) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in LIC mapping config section '{section}': "
                f"{', '.join(sorted(unknown_keys))}"
            )
        result[section] = dict(values)
    result["_meta"] = {"path": config_path, "base_dir": config_path.parent}
    return result


def resolve_config_path(value: object, *, config_path: Path | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Configured path must be a string or null, got {type(value).__name__}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and config_path is not None:
        candidate = config_path.parent / candidate
    return candidate.resolve()


__all__ = ["CONFIG_SCHEMA_VERSION", "load_yaml_config", "resolve_config_path"]
