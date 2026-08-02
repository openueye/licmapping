from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np

from lic_mapping.rosbag import (
    CameraIntrinsics,
    RosbagReader,
    SOURCE_CENTER,
    _CenteredFiveProjector,
    _SourceSlot,
    _lic2_output_geometry,
    parse_calibration,
)


def _align(buffer: bytearray, alignment: int) -> None:
    offset = len(buffer)
    aligned = 4 + ((offset - 4 + alignment - 1) & ~(alignment - 1))
    buffer.extend(b"\0" * (aligned - offset))


def _string(buffer: bytearray, value: str) -> None:
    _align(buffer, 4)
    encoded = value.encode() + b"\0"
    buffer.extend(struct.pack("<I", len(encoded)))
    buffer.extend(encoded)


def _header(timestamp_ns: int, frame_id: str) -> bytearray:
    buffer = bytearray(b"\0\1\0\0")
    _align(buffer, 4)
    buffer.extend(struct.pack("<iI", timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000))
    _string(buffer, frame_id)
    return buffer


def _image(timestamp_ns: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((8, 8, 3), 128, dtype=np.uint8))
    assert ok
    buffer = _header(timestamp_ns, "camera")
    _string(buffer, "jpeg")
    _align(buffer, 4)
    buffer.extend(struct.pack("<I", len(encoded)))
    buffer.extend(encoded.tobytes())
    return bytes(buffer)


def _odom(timestamp_ns: int, x: float) -> bytes:
    buffer = _header(timestamp_ns, "odom")
    _string(buffer, "odin1_base_link")
    for value in (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0):
        _align(buffer, 8)
        buffer.extend(struct.pack("<d", value))
    return bytes(buffer)


def _cloud(timestamp_ns: int) -> bytes:
    buffer = _header(timestamp_ns, "odom")
    _align(buffer, 4)
    buffer.extend(struct.pack("<II", 1, 3))
    fields = (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))
    buffer.extend(struct.pack("<I", len(fields)))
    for name, offset in fields:
        _string(buffer, name)
        _align(buffer, 4)
        buffer.extend(struct.pack("<I", offset))
        buffer.extend(struct.pack("<B", 6 if name == "rgb" else 7))
        _align(buffer, 4)
        buffer.extend(struct.pack("<I", 1))
    buffer.extend(struct.pack("<?", False))
    _align(buffer, 4)
    buffer.extend(struct.pack("<II", 16, 48))
    points = np.asarray(
        [(0.0, 0.0, 2.0, 0xFF0000), (0.5, 0.0, 2.0, 0x00FF00), (0.0, 0.5, 2.0, 0x0000FF)],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")],
    )
    buffer.extend(struct.pack("<I", points.nbytes))
    buffer.extend(points.tobytes())
    return bytes(buffer)


def _calibration(path: Path) -> None:
    path.write_text(
        """Tcl_0: [
1, 0, 0, 0,
0, 1, 0, 0,
0, 0, 1, 0,
0, 0, 0, 1
]
PolynomialCamera:
  A11: 4
  A12: 0
  A22: 4
  u0: 4
  v0: 4
  image_width: 8
  image_height: 8
  k2: 0
  k3: 0
  k4: 0
  k5: 0
  k6: 0
  k7: 0
""",
        encoding="utf-8",
    )


def test_rosbag_reader_builds_fixed_pose_frame(tmp_path: Path) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("relative_file_paths:\n  - data.db3\n", encoding="utf-8")
    _calibration(bag / "cam_in_ex.txt")
    database = bag / "data.db3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, topic_id INTEGER, timestamp INTEGER, data BLOB);
            """
        )
        topics = [
            (1, "/odin1/image/compressed", "sensor_msgs/msg/CompressedImage"),
            (2, "/odin1/odometry", "nav_msgs/msg/Odometry"),
            (3, "/odin1/cloud_slam", "sensor_msgs/msg/PointCloud2"),
        ]
        connection.executemany("INSERT INTO topics VALUES (?, ?, ?, 'cdr')", topics)
        connection.executemany(
            "INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)",
            [
                (1, 0, _image(0)),
                *[(1, timestamp, _image(timestamp)) for timestamp in range(500_000_000, 1_200_000_000, 100_000_000)],
                *[(2, timestamp, _odom(timestamp, 0.0)) for timestamp in range(500_000_000, 1_200_000_000, 100_000_000)],
                *[(3, timestamp, _cloud(timestamp)) for timestamp in range(500_000_000, 1_200_000_000, 100_000_000)],
            ],
        )
        connection.commit()

    reader = RosbagReader(bag, bag / "cam_in_ex.txt", max_sync_dt_ms=100.0)
    assert reader._images[0].data is None
    assert reader._clouds[0].data is None
    stream = reader.frames(limit=1)
    assert not isinstance(stream, list)
    frames = list(stream)

    assert len(frames) == 1
    assert reader.skipped_pose_frames == 1
    assert frames[0].rgb.shape == (8, 8, 3)
    assert np.isclose(frames[0].world_from_camera[0, 3], 0.0)
    assert frames[0].points_world.shape == (3, 3)
    assert np.count_nonzero(frames[0].depth_m) > 0
    assert np.all(frames[0].source_types[frames[0].depth_m > 0] == SOURCE_CENTER)


def test_lic2_output_geometry_center_crops_before_resize(tmp_path: Path) -> None:
    calibration_path = tmp_path / "cam_in_ex.txt"
    _calibration(calibration_path)
    calibration = parse_calibration(calibration_path)

    intrinsics, crop = _lic2_output_geometry(calibration, (4, 8))

    assert crop == (2, 0, 4, 8)
    assert (intrinsics.width, intrinsics.height) == (4, 8)
    assert np.allclose(
        (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy),
        (4.0, 4.0, 2.0, 4.0),
    )


def test_centered_five_rejects_conflict_but_restores_center() -> None:
    intrinsics = CameraIntrinsics(8, 8, 4.0, 4.0, 3.0, 3.0)
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    pose = np.eye(4, dtype=np.float64)
    slots = []
    for index in range(5):
        depth = 2.0 if index == 2 else (4.0 if index == 0 else 0.0)
        points = np.asarray([[0.0, 0.0, depth]], dtype=np.float32) if depth else np.empty((0, 3), dtype=np.float32)
        slots.append(_SourceSlot(index, index, rgb, pose, points))
    center, fused = _CenteredFiveProjector(intrinsics).project(tuple(slots))
    assert np.isclose(center[3, 3], 2.0)
    assert np.isclose(fused[3, 3], 2.0)

    slots[2] = _SourceSlot(2, 2, rgb, pose, np.empty((0, 3), dtype=np.float32))
    center, fused = _CenteredFiveProjector(intrinsics).project(tuple(slots))
    assert np.isclose(center[3, 3], 0.0)
    assert np.isclose(fused[3, 3], 4.0)
