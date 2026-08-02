from __future__ import annotations

import bisect
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from .rasterizer import LicCamera


IMAGE_TOPIC = "/odin1/image/compressed"
ODOMETRY_TOPIC = "/odin1/odometry"
SLAM_CLOUD_TOPIC = "/odin1/cloud_slam"
DEPTH_ADAPTER_SINGLE_CLOUD_ZBUFFER = "single_cloud_zbuffer"
SOURCE_INVALID = np.uint8(255)
SOURCE_SPNET = np.uint8(2)
SOURCE_CENTER = np.uint8(3)
SOURCE_FUSED5 = np.uint8(4)  # Retained for checkpoint compatibility.
_POINTFIELD_DTYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class BagFrame:
    """One accepted RGB/pose/world-cloud observation for mapping."""

    index: int
    timestamp_ns: int
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    world_from_camera: np.ndarray
    points_world: np.ndarray
    point_colors: np.ndarray
    point_depths_m: np.ndarray | None = None
    center_depth_m: np.ndarray | None = None
    fused5_depth_m: np.ndarray | None = None
    source_types: np.ndarray | None = None
    source_confidences: np.ndarray | None = None
    point_source_types: np.ndarray | None = None
    point_source_confidences: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected = (self.intrinsics.height, self.intrinsics.width)
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        points = np.asarray(self.points_world)
        colors = np.asarray(self.point_colors)
        pose = np.asarray(self.world_from_camera, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("BagFrame.world_from_camera must be a finite 4x4 matrix")
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("BagFrame.world_from_camera must be homogeneous")
        if self.point_depths_m is None:
            points_h = np.concatenate((points.astype(np.float64), np.ones((len(points), 1))), axis=1)
            point_depths = (np.linalg.inv(pose) @ points_h.T).T[:, 2].astype(np.float32)
        else:
            point_depths = np.asarray(self.point_depths_m)
        center_depth = depth.copy() if self.center_depth_m is None else np.asarray(self.center_depth_m)
        fused5_depth = np.zeros_like(depth) if self.fused5_depth_m is None else np.asarray(self.fused5_depth_m)
        source_types = (
            np.where(depth > 0, SOURCE_CENTER, SOURCE_INVALID).astype(np.uint8)
            if self.source_types is None
            else np.asarray(self.source_types)
        )
        source_confidences = (
            (depth > 0).astype(np.float32)
            if self.source_confidences is None
            else np.asarray(self.source_confidences)
        )
        point_source_types = (
            np.full(len(points), SOURCE_CENTER, dtype=np.uint8)
            if self.point_source_types is None
            else np.asarray(self.point_source_types)
        )
        point_source_confidences = (
            np.ones(len(points), dtype=np.float32)
            if self.point_source_confidences is None
            else np.asarray(self.point_source_confidences)
        )
        object.__setattr__(self, "center_depth_m", center_depth)
        object.__setattr__(self, "fused5_depth_m", fused5_depth)
        object.__setattr__(self, "point_depths_m", point_depths)
        object.__setattr__(self, "source_types", source_types)
        object.__setattr__(self, "source_confidences", source_confidences)
        object.__setattr__(self, "point_source_types", point_source_types)
        object.__setattr__(self, "point_source_confidences", point_source_confidences)
        if rgb.shape != (*expected, 3) or rgb.dtype != np.float32:
            raise ValueError("BagFrame.rgb must be float32 HxWx3")
        if depth.shape != expected or depth.dtype != np.float32:
            raise ValueError("BagFrame.depth_m must be float32 HxW")
        for name, value in (
            ("center_depth_m", center_depth),
            ("fused5_depth_m", fused5_depth),
            ("source_confidences", source_confidences),
        ):
            if value.shape != expected or value.dtype != np.float32:
                raise ValueError(f"BagFrame.{name} has an invalid shape or dtype")
        if source_types.shape != expected or source_types.dtype != np.uint8:
            raise ValueError("BagFrame.source_types must be uint8 HxW")
        if point_source_types.shape != (len(points),) or point_source_types.dtype != np.uint8:
            raise ValueError("BagFrame.point_source_types must be uint8 N")
        if point_source_confidences.shape != (len(points),) or point_source_confidences.dtype != np.float32:
            raise ValueError("BagFrame.point_source_confidences must be float32 N")
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32:
            raise ValueError("BagFrame.points_world must be float32 Nx3")
        if colors.shape != (len(points), 3) or colors.dtype != np.float32:
            raise ValueError("BagFrame.point_colors must be float32 Nx3")
        if point_depths.shape != (len(points),) or point_depths.dtype != np.float32:
            raise ValueError("BagFrame.point_depths_m must be float32 N")
        if not np.isfinite(rgb).all() or ((rgb < 0) | (rgb > 1)).any():
            raise ValueError("BagFrame.rgb must be finite and within [0, 1]")
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError("BagFrame.depth_m must be finite and non-negative")
        if (
            not np.isfinite(center_depth).all()
            or (center_depth < 0).any()
            or not np.isfinite(fused5_depth).all()
            or (fused5_depth < 0).any()
            or not np.isfinite(source_confidences).all()
            or (source_confidences < 0).any()
            or not np.isfinite(point_source_confidences).all()
            or (point_source_confidences < 0).any()
        ):
            raise ValueError("BagFrame source depth data must be finite and non-negative")
        if not np.isfinite(points).all() or not np.isfinite(colors).all():
            raise ValueError("BagFrame point data must be finite")
        if not np.isfinite(point_depths).all() or (point_depths < 0).any():
            raise ValueError("BagFrame.point_depths_m must be finite and non-negative")

    def lic_camera(self) -> LicCamera:
        return LicCamera(
            width=self.intrinsics.width,
            height=self.intrinsics.height,
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.cx,
            cy=self.intrinsics.cy,
            world_from_camera=torch.from_numpy(
                np.asarray(self.world_from_camera, dtype=np.float32).copy()
            ),
        )


@dataclass(frozen=True)
class Calibration:
    intrinsics: CameraIntrinsics
    t_camera_from_lidar: np.ndarray
    map_x: np.ndarray | None = None
    map_y: np.ndarray | None = None


@dataclass(frozen=True)
class _Message:
    timestamp_ns: int
    shard: Path | None = None
    row_id: int | None = None
    data: bytes | None = None


@dataclass(frozen=True)
class _SourceSlot:
    index: int
    timestamp_ns: int
    rgb: np.ndarray
    world_from_camera: np.ndarray
    points_world: np.ndarray
    point_colors: np.ndarray


@dataclass(frozen=True)
class _RejectedSlot:
    index: int
    timestamp_ns: int
    reason: str


class PoseSyncError(ValueError):
    """An image timestamp cannot be paired with a valid interpolated pose."""


def _align(offset: int, alignment: int) -> int:
    return 4 + (((offset - 4) + alignment - 1) & ~(alignment - 1))


def _u8(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<B", data, offset)[0], offset + 1


def _bool(data: bytes, offset: int) -> tuple[bool, int]:
    return struct.unpack_from("<?", data, offset)[0], offset + 1


def _i32(data: bytes, offset: int) -> tuple[int, int]:
    offset = _align(offset, 4)
    return struct.unpack_from("<i", data, offset)[0], offset + 4


def _u32(data: bytes, offset: int) -> tuple[int, int]:
    offset = _align(offset, 4)
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _f64(data: bytes, offset: int) -> tuple[float, int]:
    offset = _align(offset, 8)
    return struct.unpack_from("<d", data, offset)[0], offset + 8


def _string(data: bytes, offset: int) -> tuple[str, int]:
    offset = _align(offset, 4)
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    value = data[offset : offset + length]
    return (value[:-1].decode("utf-8") if length else ""), offset + length


def _header(data: bytes) -> tuple[int, str, int]:
    offset = 4
    sec, offset = _i32(data, offset)
    nsec, offset = _u32(data, offset)
    frame_id, offset = _string(data, offset)
    return sec * 1_000_000_000 + nsec, frame_id, offset


def _timestamp_from_prefix(data: bytes) -> int:
    if len(data) != 8:
        raise ValueError("ROS message timestamp prefix must contain 8 bytes")
    sec, nsec = struct.unpack_from("<iI", data, 0)
    return sec * 1_000_000_000 + nsec


def _parse_image(data: bytes) -> tuple[int, str, bytes]:
    timestamp, frame_id, offset = _header(data)
    _format, offset = _string(data, offset)
    size, offset = _u32(data, offset)
    return timestamp, frame_id, data[offset : offset + size]


def _parse_odometry(data: bytes) -> tuple[int, str, str, np.ndarray]:
    timestamp, frame_id, offset = _header(data)
    child_frame, offset = _string(data, offset)
    position = []
    for _ in range(3):
        value, offset = _f64(data, offset)
        position.append(value)
    quaternion = []
    for _ in range(4):
        value, offset = _f64(data, offset)
        quaternion.append(value)
    return timestamp, frame_id, child_frame, _pose_matrix(position, quaternion)


def _parse_pointcloud(data: bytes) -> dict[str, object]:
    timestamp, frame_id, offset = _header(data)
    height, offset = _u32(data, offset)
    width, offset = _u32(data, offset)
    field_count, offset = _u32(data, offset)
    fields = []
    for _ in range(field_count):
        name, offset = _string(data, offset)
        field_offset, offset = _u32(data, offset)
        datatype, offset = _u8(data, offset)
        count, offset = _u32(data, offset)
        fields.append((name, field_offset, datatype, count))
    big_endian, offset = _bool(data, offset)
    point_step, offset = _u32(data, offset)
    row_step, offset = _u32(data, offset)
    data_size, offset = _u32(data, offset)
    payload = data[offset : offset + data_size]
    if height < 1 or width < 1 or row_step != width * point_step:
        raise ValueError("PointCloud2 dimensions do not match point_step")
    if data_size != height * row_step:
        raise ValueError("PointCloud2 data size does not match row_step")
    return {
        "timestamp_ns": timestamp,
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "fields": fields,
        "big_endian": big_endian,
        "point_step": point_step,
        "data": payload,
    }


def _pose_matrix(position: list[float], quaternion_xyzw: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0:
        raise ValueError("Odometry quaternion must be non-zero")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = position
    return matrix


def _point_fields(message: dict[str, object], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    endian = ">" if message["big_endian"] else "<"
    selected = []
    for name, field_offset, datatype, count in message["fields"]:  # type: ignore[misc]
        if name not in names:
            continue
        base = _POINTFIELD_DTYPES.get(int(datatype))
        if base is None or int(count) != 1:
            raise ValueError(f"Unsupported PointCloud2 field: {name}")
        selected.append((name, np.dtype(base).newbyteorder(endian), int(field_offset)))
    missing = set(names) - {item[0] for item in selected}
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {sorted(missing)}")
    dtype = np.dtype({
        "names": [item[0] for item in selected],
        "formats": [item[1] for item in selected],
        "offsets": [item[2] for item in selected],
        "itemsize": int(message["point_step"]),
    })
    values = np.frombuffer(
        message["data"], dtype=dtype,
        count=int(message["width"]) * int(message["height"]),
    )
    return {name: np.asarray(values[name]) for name in values.dtype.names or ()}


def _decode_cloud(message: dict[str, object]) -> tuple[np.ndarray, np.ndarray | None]:
    names = ("x", "y", "z")
    fields = _point_fields(message, names)
    points = np.stack([fields[name] for name in names], axis=1).astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    colors: np.ndarray | None = None
    field_names = {item[0] for item in message["fields"]}  # type: ignore[misc]
    if "rgb" in field_names:
        rgb_field = _point_fields(message, ("rgb",))["rgb"]
        if np.issubdtype(rgb_field.dtype, np.floating):
            packed = rgb_field.astype(np.float32, copy=False).view(np.uint32)
        else:
            packed = rgb_field.astype(np.uint32, copy=False)
        colors = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1)
        colors = colors.astype(np.float32) / 255.0
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    return points, colors


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    left = q0 / np.linalg.norm(q0)
    right = q1 / np.linalg.norm(q1)
    dot = float(np.dot(left, right))
    if dot < 0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1, 1))
    if dot > 0.9995:
        result = left + alpha * (right - left)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    return (
        np.sin((1 - alpha) * angle) / np.sin(angle) * left
        + np.sin(alpha * angle) / np.sin(angle) * right
    )


class PoseTrack:
    def __init__(self, samples: list[tuple[int, np.ndarray]]) -> None:
        if len(samples) < 2:
            raise ValueError("At least two odometry samples are required")
        samples = sorted(samples, key=lambda item: item[0])
        if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
            raise ValueError("Odometry timestamps must be strictly increasing")
        self._timestamps = [item[0] for item in samples]
        self._poses = [item[1] for item in samples]

    def interpolate(self, timestamp_ns: int, *, max_dt_ns: int) -> np.ndarray:
        index = bisect.bisect_left(self._timestamps, int(timestamp_ns))
        if index < len(self._timestamps) and self._timestamps[index] == timestamp_ns:
            return self._poses[index].copy()
        if index == len(self._timestamps) and self._timestamps[-1] == timestamp_ns:
            return self._poses[-1].copy()
        if index == 0 or index == len(self._timestamps):
            raise PoseSyncError("Image timestamp is outside odometry coverage")
        left, right = index - 1, index
        dt_left = timestamp_ns - self._timestamps[left]
        dt_right = self._timestamps[right] - timestamp_ns
        if min(dt_left, dt_right) > max_dt_ns or self._timestamps[right] - self._timestamps[left] > 2 * max_dt_ns:
            raise PoseSyncError("Image timestamp has no valid odometry bracket")
        alpha = dt_left / (self._timestamps[right] - self._timestamps[left])
        q0 = _matrix_to_quaternion(self._poses[left][:3, :3])
        q1 = _matrix_to_quaternion(self._poses[right][:3, :3])
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _pose_matrix([0, 0, 0], _slerp(q0, q1, alpha))[:3, :3]
        pose[:3, 3] = (1 - alpha) * self._poses[left][:3, 3] + alpha * self._poses[right][:3, 3]
        return pose

    def nearest(self, timestamp_ns: int, *, max_dt_ns: int) -> np.ndarray:
        """Return one recorded pose inside LIC's strict synchronization window."""
        index = bisect.bisect_left(self._timestamps, int(timestamp_ns))
        candidates = self._timestamps[max(0, index - 1) : min(len(self._timestamps), index + 1)]
        if not candidates:
            raise PoseSyncError("Image timestamp has no recorded odometry sample")
        selected_timestamp = min(candidates, key=lambda value: abs(value - timestamp_ns))
        if abs(selected_timestamp - timestamp_ns) > max_dt_ns:
            raise PoseSyncError("Image timestamp has no synchronized odometry sample")
        return self._poses[self._timestamps.index(selected_timestamp)].copy()


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = 2 * np.sqrt(max(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12))
            w, x, y, z = (matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = 2 * np.sqrt(max(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12))
            w, x, y, z = (matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2 * np.sqrt(max(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12))
            w, x, y, z = (matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s
    return np.asarray([x, y, z, w], dtype=np.float64)


def parse_calibration(path: Path) -> Calibration:
    matrix_name: str | None = None
    matrix_values: list[float] = []
    in_matrix = False
    section: str | None = None
    params: dict[str, float | str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(": ["):
            matrix_name, in_matrix = line[:-3], True
            matrix_values = []
            continue
        if in_matrix:
            if line == "]":
                in_matrix = False
            else:
                matrix_values.extend(float(item.strip()) for item in line.rstrip(",").split(",") if item.strip())
            continue
        if line.endswith(":") and not raw.startswith(" "):
            section = line[:-1]
            continue
        if section and ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            value = value.split("#", 1)[0].strip()
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
    required = ("A11", "A22", "u0", "v0", "image_width", "image_height")
    if matrix_name is None or len(matrix_values) != 16 or any(name not in params for name in required):
        raise ValueError(f"Invalid camera calibration: {path}")
    width, height = int(params["image_width"]), int(params["image_height"])
    fx = float(params["A11"])
    fy = float(params["A22"])
    cx = float(params["u0"])
    cy = float(params["v0"])
    map_x, map_y = _fishpoly_maps(
        width, height, float(params["A11"]), float(params.get("A12", 0.0)),
        float(params["A22"]), float(params["u0"]), float(params["v0"]),
        [float(params.get(f"k{i}", 0.0)) for i in range(2, 8)],
    )
    return Calibration(
        CameraIntrinsics(width, height, fx, fy, cx, cy),
        np.asarray(matrix_values, dtype=np.float64).reshape(4, 4),
        map_x,
        map_y,
    )


def _fishpoly_maps(width: int, height: int, a11: float, a12: float, a22: float, u0: float, v0: float, coefficients: list[float]) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    y = (v - v0) / a22
    x = (u - u0 - a12 * y) / a11
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan(radius)
    theta_distorted = theta.copy()
    for power, coefficient in enumerate(coefficients, start=2):
        theta_distorted += coefficient * theta**power
    scale = np.divide(theta_distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
    return (a11 * x * scale + a12 * y * scale + u0).astype(np.float32), (a22 * y * scale + v0).astype(np.float32)


def _output_geometry(
    calibration: Calibration,
    output_size: tuple[int, int],
) -> tuple[CameraIntrinsics, tuple[int, int, int, int]]:
    """Apply the configured center crop/resize and update intrinsics."""

    width, height = output_size
    source_width = calibration.intrinsics.width
    source_height = calibration.intrinsics.height
    target_aspect = width / height
    crop_width = source_width
    crop_height = int(round(crop_width / target_aspect))
    if crop_height > source_height:
        crop_height = source_height
        crop_width = int(round(crop_height * target_aspect))
    crop_x = (source_width - crop_width) // 2
    crop_y = (source_height - crop_height) // 2
    scale_x = width / crop_width
    scale_y = height / crop_height
    intrinsics = CameraIntrinsics(
        width,
        height,
        calibration.intrinsics.fx * scale_x,
        calibration.intrinsics.fy * scale_y,
        (calibration.intrinsics.cx - crop_x) * scale_x,
        (calibration.intrinsics.cy - crop_y) * scale_y,
    )
    return intrinsics, (crop_x, crop_y, crop_width, crop_height)


def _bag_shards(root: Path) -> tuple[Path, ...]:
    metadata = root / "metadata.yaml"
    if not metadata.is_file():
        raise ValueError(f"Missing ROSBAG metadata.yaml: {metadata}")
    declared = tuple(match.group(1) for match in re.finditer(r"^\s*-\s*['\"]?([^'\"\s]+\.db3)['\"]?\s*$", metadata.read_text(), re.MULTILINE))
    shards = tuple((root / name).resolve() for name in declared) if declared else tuple(sorted(root.glob("*.db3")))
    if not shards or any(path.parent != root or not path.is_file() for path in shards):
        raise ValueError("ROSBAG must contain valid .db3 shards")
    return shards


class RosbagReader:
    """Read fixed-pose Odin frames using a single synchronized SLAM cloud.

    The reference Gaussian-LIC executable receives a depth image from its
    upstream ROS frontend.  Odin ROS2 bags do not contain that topic, so this
    reader explicitly uses the approved experimental adapter: the matched
    ``/odin1/cloud_slam`` cloud is z-buffer projected into the matched image.
    It never mixes points from neighbouring frames.
    """

    def __init__(
        self,
        rosbag_dir: Path,
        calibration_path: Path,
        *,
        max_sync_dt_ms: float = 10.0,
        max_cloud_dt_ms: float = 10.0,
        resize: tuple[int, int] | None = None,
        depth_adapter: str = DEPTH_ADAPTER_SINGLE_CLOUD_ZBUFFER,
    ) -> None:
        if max_sync_dt_ms <= 0 or max_cloud_dt_ms <= 0:
            raise ValueError("max_sync_dt_ms and max_cloud_dt_ms must be positive")
        if resize is not None and (len(resize) != 2 or resize[0] < 1 or resize[1] < 1):
            raise ValueError("resize must contain positive width and height")
        if depth_adapter != DEPTH_ADAPTER_SINGLE_CLOUD_ZBUFFER:
            raise ValueError(f"Unsupported experimental depth adapter: {depth_adapter}")
        self.root = Path(rosbag_dir).resolve()
        self.calibration = parse_calibration(Path(calibration_path).resolve())
        output_size = resize or (
            self.calibration.intrinsics.width,
            self.calibration.intrinsics.height,
        )
        self.intrinsics, self._crop = _output_geometry(self.calibration, output_size)
        self._resize = resize
        self._depth_adapter = depth_adapter
        self._images: list[_Message] = []
        self._clouds: list[_Message] = []
        poses: list[tuple[int, np.ndarray]] = []
        for shard in _bag_shards(self.root):
            with sqlite3.connect(str(shard)) as connection:
                topics = {str(name): (int(topic_id), str(message_type), str(serialization)) for topic_id, name, message_type, serialization in connection.execute("SELECT id, name, type, serialization_format FROM topics")}
                for topic in (IMAGE_TOPIC, ODOMETRY_TOPIC, SLAM_CLOUD_TOPIC):
                    if topic not in topics:
                        continue
                    topic_id, message_type, serialization = topics[topic]
                    expected = {IMAGE_TOPIC: "sensor_msgs/msg/CompressedImage", ODOMETRY_TOPIC: "nav_msgs/msg/Odometry", SLAM_CLOUD_TOPIC: "sensor_msgs/msg/PointCloud2"}[topic]
                    if message_type != expected or serialization != "cdr":
                        raise ValueError(f"Unexpected topic identity for {topic}")
                    query = (
                        "SELECT rowid, data FROM messages WHERE topic_id = ? ORDER BY rowid"
                        if topic == ODOMETRY_TOPIC
                        else "SELECT rowid, substr(data, 5, 8) FROM messages WHERE topic_id = ? ORDER BY rowid"
                    )
                    for row_id, payload in connection.execute(query, (topic_id,)):
                        if topic == IMAGE_TOPIC:
                            timestamp = _timestamp_from_prefix(bytes(payload))
                            self._images.append(_Message(timestamp, shard=shard, row_id=int(row_id)))
                        elif topic == SLAM_CLOUD_TOPIC:
                            timestamp = _timestamp_from_prefix(bytes(payload))
                            self._clouds.append(_Message(timestamp, shard=shard, row_id=int(row_id)))
                        else:
                            data = bytes(payload)
                            timestamp = _header(data)[0]
                            _stamp, _frame, _child, pose = _parse_odometry(data)
                            poses.append((timestamp, pose))
        if not self._images or not self._clouds:
            raise ValueError("ROSBAG must contain image and /odin1/cloud_slam messages")
        self._images.sort(key=lambda item: item.timestamp_ns)
        self._clouds.sort(key=lambda item: item.timestamp_ns)
        self._cloud_timestamps = [item.timestamp_ns for item in self._clouds]
        self._poses = PoseTrack(poses)
        self._max_sync_dt_ns = int(round(max_sync_dt_ms * 1_000_000))
        self._max_cloud_dt_ns = int(round(max_cloud_dt_ms * 1_000_000))
        self._lidar_from_camera = np.linalg.inv(self.calibration.t_camera_from_lidar)
        self._skipped_pose_frames = 0
        self._rejected_source_frames = 0

    def __len__(self) -> int:
        return len(self._images)

    @property
    def skipped_pose_frames(self) -> int:
        """Number of image frames skipped by the most recent frames() call."""
        return self._skipped_pose_frames

    @property
    def rejected_source_frames(self) -> int:
        """Number of source frames rejected for missing or unsynchronized clouds."""
        return self._rejected_source_frames

    @property
    def input_adapter(self) -> dict[str, object]:
        return {
            "depth": self._depth_adapter,
            "depth_alignment": "experimental_adapter",
            "sync_window_ms": self._max_sync_dt_ns / 1_000_000,
            "cloud_window_ms": self._max_cloud_dt_ns / 1_000_000,
            "neighboring_frame_fusion": False,
        }

    def frames(self, *, start: int = 0, limit: int | None = None) -> Iterator[BagFrame]:
        """Stream accepted single-frame mapping observations."""
        return self.iter_frames(start=start, limit=limit)

    def iter_frames(self, *, start: int = 0, limit: int | None = None) -> Iterator[BagFrame]:
        if start < 0 or (limit is not None and limit < 1):
            raise ValueError("start must be non-negative and limit must be positive")
        self._skipped_pose_frames = 0
        self._rejected_source_frames = 0
        emitted = 0
        for index, image in enumerate(self._images[start:], start=start):
            slot = self._build_source_slot(index, image)
            if isinstance(slot, _RejectedSlot):
                if slot.reason == "pose":
                    self._skipped_pose_frames += 1
                else:
                    self._rejected_source_frames += 1
                continue
            yield self._build_frame(slot)
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    @staticmethod
    def _message_data(message: _Message) -> bytes:
        if message.data is not None:
            return message.data
        if message.shard is None or message.row_id is None:
            raise ValueError("Message has no backing ROSBAG row")
        with sqlite3.connect(str(message.shard)) as connection:
            row = connection.execute("SELECT data FROM messages WHERE rowid = ?", (message.row_id,)).fetchone()
        if row is None:
            raise ValueError(f"ROSBAG message row disappeared: {message.shard}:{message.row_id}")
        return bytes(row[0])

    def _build_source_slot(self, index: int, image: _Message) -> _SourceSlot | _RejectedSlot:
        timestamp, _frame_id, compressed = _parse_image(self._message_data(image))
        bgr = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image {index}")
        if self.calibration.map_x is not None:
            bgr = cv2.remap(bgr, self.calibration.map_x, self.calibration.map_y, cv2.INTER_LINEAR)
        crop_x, crop_y, crop_width, crop_height = self._crop
        bgr = bgr[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self._resize is not None:
            rgb = cv2.resize(rgb, self._resize, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
        try:
            world_from_base = self._poses.nearest(timestamp, max_dt_ns=self._max_sync_dt_ns)
        except PoseSyncError:
            return _RejectedSlot(index, timestamp, "pose")
        world_from_camera = world_from_base @ self._lidar_from_camera
        right = bisect.bisect_left(self._cloud_timestamps, timestamp)
        candidates = self._clouds[max(0, right - 1) : min(len(self._clouds), right + 1)]
        if not candidates:
            return _RejectedSlot(index, timestamp, "cloud")
        cloud = min(candidates, key=lambda item: abs(item.timestamp_ns - timestamp))
        if abs(cloud.timestamp_ns - timestamp) > self._max_cloud_dt_ns:
            return _RejectedSlot(index, timestamp, "cloud")
        cloud_message = _parse_pointcloud(self._message_data(cloud))
        points, colors = _decode_cloud(cloud_message)
        if colors is None:
            return _RejectedSlot(index, timestamp, "cloud")
        return _SourceSlot(index, timestamp, rgb, world_from_camera, points, colors)

    def _build_frame(self, slot: _SourceSlot) -> BagFrame:
        depth, visible_indices = _project_visible_cloud(
            slot.points_world,
            slot.world_from_camera,
            self.intrinsics,
            min_depth=0.1,
            max_depth=200.0,
        )
        visible_points = slot.points_world[visible_indices]
        visible_colors = slot.point_colors[visible_indices]
        source_types = np.where(depth > 0, SOURCE_CENTER, SOURCE_INVALID).astype(np.uint8)
        source_confidences = (depth > 0).astype(np.float32)
        return BagFrame(
            slot.index,
            slot.timestamp_ns,
            slot.rgb,
            depth,
            self.intrinsics,
            slot.world_from_camera,
            visible_points,
            visible_colors,
            center_depth_m=depth,
            source_types=source_types,
            source_confidences=source_confidences,
            point_source_types=np.full(len(visible_points), SOURCE_CENTER, dtype=np.uint8),
            point_source_confidences=np.ones(len(visible_points), dtype=np.float32),
        )


def _project_visible_cloud(
    points_world: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer a cloud and return exactly the visible point/color rows.

    The returned indices select one nearest source point for each non-zero
    depth pixel, so map initialization and depth supervision share one data
    contract. Ties are broken by original cloud row for deterministic replay.
    """
    camera_from_world = np.linalg.inv(world_from_camera)
    points_h = np.concatenate((points_world.astype(np.float64), np.ones((len(points_world), 1))), axis=1)
    camera = (camera_from_world @ points_h.T).T[:, :3]
    z = camera[:, 2]
    valid = (
        np.isfinite(camera).all(axis=1)
        & (z >= min_depth)
        & (z <= max_depth)
    )
    projected_u = intrinsics.fx * camera[valid, 0] / z[valid] + intrinsics.cx
    projected_v = intrinsics.fy * camera[valid, 1] / z[valid] + intrinsics.cy
    rows = np.rint(projected_v).astype(np.int64)
    cols = np.rint(projected_u).astype(np.int64)
    inside = (
        np.isfinite(projected_u)
        & np.isfinite(projected_v)
        & (rows >= 0)
        & (rows < intrinsics.height)
        & (cols >= 0)
        & (cols < intrinsics.width)
    )
    source_indices = np.flatnonzero(valid)[inside]
    rows, cols, projected_z = rows[inside], cols[inside], z[valid][inside].astype(np.float32)
    if len(source_indices) == 0:
        return np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32), source_indices
    pixel_indices = rows * intrinsics.width + cols
    order = np.lexsort((source_indices, projected_z))
    sorted_pixels = pixel_indices[order]
    first_per_pixel = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    winners = order[first_per_pixel]
    zbuffer = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)
    zbuffer[rows[winners], cols[winners]] = projected_z[winners]
    return zbuffer, source_indices[winners]


def _project_depth(
    points_world: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Compatibility helper returning only the z-buffer depth image."""
    depth, _ = _project_visible_cloud(
        points_world,
        world_from_camera,
        intrinsics,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return depth
