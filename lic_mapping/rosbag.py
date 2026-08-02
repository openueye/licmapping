from __future__ import annotations

import bisect
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .rasterizer import LicCamera


IMAGE_TOPIC = "/odin1/image/compressed"
ODOMETRY_TOPIC = "/odin1/odometry"
SLAM_CLOUD_TOPIC = "/odin1/cloud_slam"
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

    def __post_init__(self) -> None:
        expected = (self.intrinsics.height, self.intrinsics.width)
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        points = np.asarray(self.points_world)
        colors = np.asarray(self.point_colors)
        if rgb.shape != (*expected, 3) or rgb.dtype != np.float32:
            raise ValueError("BagFrame.rgb must be float32 HxWx3")
        if depth.shape != expected or depth.dtype != np.float32:
            raise ValueError("BagFrame.depth_m must be float32 HxW")
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32:
            raise ValueError("BagFrame.points_world must be float32 Nx3")
        if colors.shape != (len(points), 3) or colors.dtype != np.float32:
            raise ValueError("BagFrame.point_colors must be float32 Nx3")
        if not np.isfinite(rgb).all() or ((rgb < 0) | (rgb > 1)).any():
            raise ValueError("BagFrame.rgb must be finite and within [0, 1]")
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError("BagFrame.depth_m must be finite and non-negative")
        if not np.isfinite(points).all() or not np.isfinite(colors).all():
            raise ValueError("BagFrame point data must be finite")
        pose = np.asarray(self.world_from_camera, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("BagFrame.world_from_camera must be a finite 4x4 matrix")
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("BagFrame.world_from_camera must be homogeneous")

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
    data: bytes


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
        if index == 0 or index == len(self._timestamps):
            raise ValueError("Image timestamp is outside odometry coverage")
        if self._timestamps[index] == timestamp_ns:
            return self._poses[index].copy()
        left, right = index - 1, index
        dt_left = timestamp_ns - self._timestamps[left]
        dt_right = self._timestamps[right] - timestamp_ns
        if min(dt_left, dt_right) > max_dt_ns or self._timestamps[right] - self._timestamps[left] > 2 * max_dt_ns:
            raise ValueError("Image timestamp has no valid odometry bracket")
        alpha = dt_left / (self._timestamps[right] - self._timestamps[left])
        q0 = _matrix_to_quaternion(self._poses[left][:3, :3])
        q1 = _matrix_to_quaternion(self._poses[right][:3, :3])
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _pose_matrix([0, 0, 0], _slerp(q0, q1, alpha))[:3, :3]
        pose[:3, 3] = (1 - alpha) * self._poses[left][:3, 3] + alpha * self._poses[right][:3, 3]
        return pose


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
    fx, fy = width / 2.0, height / 2.0
    map_x, map_y = _fishpoly_maps(
        width, height, float(params["A11"]), float(params.get("A12", 0.0)),
        float(params["A22"]), float(params["u0"]), float(params["v0"]),
        [float(params.get(f"k{i}", 0.0)) for i in range(2, 8)],
    )
    return Calibration(
        CameraIntrinsics(width, height, fx, fy, width / 2.0, height / 2.0),
        np.asarray(matrix_values, dtype=np.float64).reshape(4, 4),
        map_x,
        map_y,
    )


def _fishpoly_maps(width: int, height: int, a11: float, a12: float, a22: float, u0: float, v0: float, coefficients: list[float]) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x, y = (u - width / 2) / (width / 2), (v - height / 2) / (height / 2)
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan(radius)
    theta_distorted = theta.copy()
    for power, coefficient in enumerate(coefficients, start=2):
        theta_distorted += coefficient * theta**power
    scale = np.divide(theta_distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
    return (a11 * x * scale + a12 * y * scale + u0).astype(np.float32), (a22 * y * scale + v0).astype(np.float32)


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
    """Read a finite Odin ROS2 bag into fixed-pose RGB/world-cloud frames."""

    def __init__(self, rosbag_dir: Path, calibration_path: Path, *, max_sync_dt_ms: float = 50.0, resize: tuple[int, int] | None = None) -> None:
        if max_sync_dt_ms <= 0:
            raise ValueError("max_sync_dt_ms must be positive")
        if resize is not None and (len(resize) != 2 or resize[0] < 1 or resize[1] < 1):
            raise ValueError("resize must contain positive width and height")
        self.root = Path(rosbag_dir).resolve()
        self.calibration = parse_calibration(Path(calibration_path).resolve())
        if resize is None:
            self.intrinsics = self.calibration.intrinsics
        else:
            width, height = resize
            scale_x = width / self.calibration.intrinsics.width
            scale_y = height / self.calibration.intrinsics.height
            self.intrinsics = CameraIntrinsics(
                width,
                height,
                self.calibration.intrinsics.fx * scale_x,
                self.calibration.intrinsics.fy * scale_y,
                self.calibration.intrinsics.cx * scale_x,
                self.calibration.intrinsics.cy * scale_y,
            )
        self._resize = resize
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
                    for row_id, payload in connection.execute("SELECT rowid, data FROM messages WHERE topic_id = ? ORDER BY rowid", (topic_id,)):
                        data = bytes(payload)
                        timestamp = _header(data)[0]
                        message = _Message(timestamp, data)
                        if topic == IMAGE_TOPIC:
                            self._images.append(message)
                        elif topic == SLAM_CLOUD_TOPIC:
                            self._clouds.append(message)
                        else:
                            _stamp, _frame, _child, pose = _parse_odometry(data)
                            poses.append((timestamp, pose))
        if not self._images or not self._clouds:
            raise ValueError("ROSBAG must contain image and /odin1/cloud_slam messages")
        self._images.sort(key=lambda item: item.timestamp_ns)
        self._clouds.sort(key=lambda item: item.timestamp_ns)
        self._cloud_timestamps = [item.timestamp_ns for item in self._clouds]
        self._poses = PoseTrack(poses)
        self._max_sync_dt_ns = int(round(max_sync_dt_ms * 1_000_000))
        self._lidar_from_camera = np.linalg.inv(self.calibration.t_camera_from_lidar)

    def __len__(self) -> int:
        return len(self._images)

    def frames(self, *, start: int = 0, limit: int | None = None) -> list[BagFrame]:
        if start < 0 or (limit is not None and limit < 1):
            raise ValueError("start must be non-negative and limit must be positive")
        selected = self._images[start:] if limit is None else self._images[start : start + limit]
        result = []
        for index, image in enumerate(selected, start=start):
            result.append(self._build_frame(index, image))
        return result

    def _build_frame(self, index: int, image: _Message) -> BagFrame:
        timestamp, _frame_id, compressed = _parse_image(image.data)
        bgr = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image {index}")
        if self.calibration.map_x is not None:
            bgr = cv2.remap(bgr, self.calibration.map_x, self.calibration.map_y, cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self._resize is not None:
            rgb = cv2.resize(rgb, self._resize, interpolation=cv2.INTER_AREA).astype(np.float32)
        world_from_base = self._poses.interpolate(timestamp, max_dt_ns=self._max_sync_dt_ns)
        world_from_camera = world_from_base @ self._lidar_from_camera
        right = bisect.bisect_left(self._cloud_timestamps, timestamp)
        candidates = self._clouds[max(0, right - 1) : min(len(self._clouds), right + 1)]
        cloud = min(candidates, key=lambda item: abs(item.timestamp_ns - timestamp))
        if abs(cloud.timestamp_ns - timestamp) > self._max_sync_dt_ns:
            raise ValueError(f"No cloud timestamp near image {index}")
        cloud_message = _parse_pointcloud(cloud.data)
        points, cloud_colors = _decode_cloud(cloud_message)
        depth, sampled_colors, visible = _project_cloud(points, rgb, world_from_camera, self.intrinsics)
        if cloud_colors is None:
            points, sampled_colors = points[visible], sampled_colors[visible]
        else:
            sampled_colors = cloud_colors
        return BagFrame(index, timestamp, rgb, depth, self.intrinsics, world_from_camera, points, sampled_colors)


def _project_cloud(points: np.ndarray, rgb: np.ndarray, world_from_camera: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_from_world = np.linalg.inv(world_from_camera)
    points_h = np.concatenate((points.astype(np.float64), np.ones((len(points), 1))), axis=1)
    camera = (camera_from_world @ points_h.T).T[:, :3]
    z = camera[:, 2]
    u = intrinsics.fx * camera[:, 0] / np.maximum(z, 1e-8) + intrinsics.cx
    v = intrinsics.fy * camera[:, 1] / np.maximum(z, 1e-8) + intrinsics.cy
    valid = np.isfinite(camera).all(axis=1) & (z > 0.1) & (u >= 0) & (u < intrinsics.width - 1) & (v >= 0) & (v < intrinsics.height - 1)
    rows = np.clip(np.rint(v[valid]).astype(np.int64), 0, intrinsics.height - 1)
    cols = np.clip(np.rint(u[valid]).astype(np.int64), 0, intrinsics.width - 1)
    sampled = rgb[rows, cols]
    depth = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)
    for row, col, value in zip(rows, cols, z[valid].astype(np.float32)):
        current = depth[row, col]
        if current == 0 or value < current:
            depth[row, col] = value
    visible = np.zeros(len(points), dtype=bool)
    visible[np.flatnonzero(valid)] = True
    return depth, sampled.astype(np.float32), visible
